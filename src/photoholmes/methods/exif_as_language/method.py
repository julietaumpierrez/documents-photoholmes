# Derived from https://github.com/hellomuffin/exif-as-language
import random
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.methods.base import BaseMethod
from photoholmes.methods.exif_as_language.clip import ClipModel
from photoholmes.utils.patched_image import PatchedImage
from photoholmes.utils.pca import PCA

from .utils import cosine_similarity, mean_shift, normalized_cut


# FIXME fix docstrings
class EXIFAsLanguage(BaseMethod):
    def __init__(
        self,
        transformer: Literal["distilbert"],
        visual: Literal["resnet50"],
        patch_size: int = 128,
        num_per_dim: int = 30,
        feat_batch_size: int = 32,
        pred_batch_size: int = 1024,
        device: str = "cuda:0",
        ms_window: int = 10,
        ms_iter: int = 5,
        pooling: Literal["cls", "mean"] = "mean",
        state_dict_path: Optional[str] = None,
        seed: int = 44,
    ):
        """
        Parameters
        ----------
        transformer: Transformer used for text embedding
        vision: Vision model used for image embedding
        patch_size : int, optional
            Size of patches, by default 128
        num_per_dim : int, optional
            Number of patches to use along the largest dimension,
            by default None (stride using patch_size)
        device : str, optional
            , by default "cuda:0"
        ms_window: Window size for mean shift
        ms_iter: Number of iterations for mean shift
        state_dict_paths: Path to weights
        """
        random.seed(seed)
        super().__init__()

        clipNet = ClipModel(vision=visual, text=transformer, pooling=pooling)
        if state_dict_path:
            checkpoint = torch.load(state_dict_path, map_location=device)
            clipNet.load_state_dict(checkpoint)

        self.patch_size = patch_size
        self.num_per_dim = num_per_dim
        self.feat_batch_size = feat_batch_size
        self.pred_batch_size = pred_batch_size
        self.device = torch.device(device)
        self.ms_window, self.ms_iter = ms_window, ms_iter
        self.net = clipNet

        self.net.eval()
        self.net.to(device)

    def predict(
        self,
        image: Tensor,
        original_image_size: Tuple[int, int],
    ):
        """
        Parameters
        ----------
        img : torch.Tensor
            [C, H, W], range: [0, 1]
        original_image_size : Tuple[int, int]
            [H, W]

        Returns
        -------
        Dict[str, Any]
            ms : np.ndarray (float32)
                Consistency map, [H, W], range [0, 1]
            ncuts : np.ndarray (float32)
                Localization map, [H, W], range [0, 1]
            score : float
                Prediction score, higher indicates existence of manipulation
        """

        # Initialize image and attributes
        height, width = original_image_size
        p_img = self.init_img(image)
        # Precompute features for each patch
        with torch.no_grad():
            patch_features = self.get_patch_feats(
                p_img, batch_size=self.feat_batch_size
            )

        # PCA visualization
        pca = PCA(n_components=3, whiten=True)
        feature_transform = pca.fit_transform(patch_features.cpu().numpy())
        pred_pca_map = self.predict_pca_map(
            p_img, feature_transform, batch_size=self.pred_batch_size
        )

        # Predict consistency maps
        pred_maps = self.predict_consistency_maps(
            p_img, patch_features, batch_size=self.pred_batch_size
        ).numpy()

        # Produce a single response map
        ms = mean_shift(
            pred_maps.reshape((-1, pred_maps.shape[0] * pred_maps.shape[1])),
            pred_maps,
            window=self.ms_window,
            iter=self.ms_iter,
        )

        # Run clustering to get localization map
        ncuts = normalized_cut(pred_maps)
        # TODO: change resize to our own implementation
        out_ms = cv2.resize(ms, (width, height), interpolation=cv2.INTER_LINEAR)
        out_ncuts = cv2.resize(
            ncuts.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        out_pca = np.zeros((height, width, 3))
        p1, p3 = np.percentile(pred_pca_map, 0.5), np.percentile(pred_pca_map, 99.5)
        pred_pca_map = (pred_pca_map - p1) / (p3 - p1) * 255  # >0
        pred_pca_map[pred_pca_map < 0] = 0
        pred_pca_map[pred_pca_map > 255] = 255
        for i in range(3):
            out_pca[:, :, i] = cv2.resize(
                pred_pca_map[:, :, i], (width, height), interpolation=cv2.INTER_LINEAR
            )
        score = pred_maps.mean()
        affinity_matrix = self.generate_afinity_matrix(patch_features)

        return {
            "heatmap": out_ms,
            "mask": out_ncuts,
            "score": score,
            "pred_maps": pred_maps,
            "pca": out_pca,
            "affinity_matrix": affinity_matrix,
        }

    def init_img(self, img: Tensor) -> PatchedImage:
        # Initialize image and attributes
        _, height, width = img.shape
        assert (
            min(height, width) > self.patch_size
        ), "Image must be bigger than patch size"
        img = img.to(self.device)
        p_img = PatchedImage(img, self.patch_size, num_per_dim=self.num_per_dim)

        return p_img

    def predict_consistency_maps(
        self, img: PatchedImage, patch_features: Tensor, batch_size=64
    ):
        # For each patch, how many overlapping patches?
        spread = max(1, img.patch_size // img.stride)

        # Aggregate prediction maps; for each patch, compared to each other patch
        responses = torch.zeros(
            (
                img.max_h_idx + spread - 1,
                img.max_w_idx + spread - 1,
                img.max_h_idx + spread - 1,
                img.max_w_idx + spread - 1,
            )
        )
        # Number of predictions for each patch
        vote_counts = (
            torch.zeros(
                (
                    img.max_h_idx + spread - 1,
                    img.max_w_idx + spread - 1,
                    img.max_h_idx + spread - 1,
                    img.max_w_idx + spread - 1,
                )
            )
            + 1e-4
        )

        # Perform prediction
        for idxs in img.pred_idxs_gen(batch_size=batch_size):
            # a to be compared to b
            patch_a_idxs = idxs[:, :2]  # [B, 2]
            patch_b_idxs = idxs[:, 2:]  # [B, 2]

            # Convert 2D index into its 1D version
            a_idxs = np.ravel_multi_index(
                patch_a_idxs.T.tolist(), [img.max_h_idx, img.max_w_idx]
            )  # [B]
            b_idxs = np.ravel_multi_index(
                patch_b_idxs.T.tolist(), [img.max_h_idx, img.max_w_idx]
            )

            # Grab corresponding features
            a_feats = patch_features[a_idxs]  # [B, 4096]
            b_feats = patch_features[b_idxs]

            sim = self.patch_similarity(a_feats, b_feats)

            # FIXME Is it possible to vectorize this?
            # Accumulate predictions for overlapping patches
            for i in range(len(sim)):
                responses[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += sim[i]
                vote_counts[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += 1

        # Normalize predictions
        return responses / vote_counts

    def predict_pca_map(
        self, img: PatchedImage, patch_features: NDArray, batch_size=64
    ) -> NDArray:
        # For each patch, how many overlapping patches?
        spread = max(1, img.patch_size // img.stride)

        # Aggregate prediction maps; for each patch, compared to each other patch
        responses = np.zeros(
            (img.max_h_idx + spread - 1, img.max_w_idx + spread - 1, 3)
        )
        # Number of predictions for each patch
        vote_counts = (
            np.zeros((img.max_h_idx + spread - 1, img.max_w_idx + spread - 1, 3)) + 1e-4
        )

        # Perform prediction
        for idxs in img.idxs_gen(batch_size=batch_size):
            # a to be compared to b
            patch_a_idxs = idxs[:, :2]  # [B, 2]

            # Convert 2D index into its 1D version
            a_idxs = np.ravel_multi_index(
                patch_a_idxs.T.tolist(), [img.max_h_idx, img.max_w_idx]
            )  # [B]

            # Grab corresponding features
            a_feats = patch_features[a_idxs]  # [B, 3]

            # FIXME Is it possible to vectorize this?
            # Accumulate predictions for overlapping patches
            for i in range(a_feats.shape[0]):
                responses[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    :,
                ] += a_feats[i]
                vote_counts[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    :,
                ] += 1

        # Normalize predictions
        return responses / vote_counts

    def patch_similarity(self, a_feats: Tensor, b_feats: Tensor) -> Tensor:
        cos = cosine_similarity(a_feats, b_feats).diagonal()
        cos = 1 - cos
        cos = cos.cpu()
        return cos

    def get_patch_feats(self, img: PatchedImage, batch_size=32):
        """
        Get features for every patch in the image.
        Features used to compute if two patches share the same EXIF attributes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to be fed into the network, by default 32

        Returns
        -------
        torch.Tensor
            [n_patches, 4096]
        """
        # Compute feature vector for each image patch
        patch_features = []

        # Generator for patches; raster scan order
        for patches in img.patches_gen(batch_size):
            processed_patches = patches.to(self.device)
            feat = self.net.encode_image(processed_patches)

            if len(feat.shape) == 1:
                feat = feat.view(1, -1)
            patch_features.append(feat)

        # [n_patches, n_features]
        patch_features = torch.cat(patch_features, dim=0)

        return patch_features

    def generate_afinity_matrix(self, patch_features: Tensor) -> Tensor:
        patch_features = torch.nn.functional.normalize(patch_features)
        result = torch.matmul(patch_features, patch_features.t())

        return result

    def get_valid_patch_mask(self, mask: PatchedImage, batch_size=32):
        valid_mask = []
        for patches in mask.patches_gen(batch_size):
            patches = patches.reshape(
                patches.shape[0], -1
            )  # [batch_size, patch_size * patch_size]
            patches_sum = torch.sum(patches, dim=1)  # [batch_size]
            positive_mask = patches_sum > self.patch_size * self.patch_size * 0.9
            negative_mask = patches_sum == 0
            valid_mask.append(positive_mask.long() - negative_mask.long())
        valid_mask = torch.cat(valid_mask, dim=0)
        return valid_mask
