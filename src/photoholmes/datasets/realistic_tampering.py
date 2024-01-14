import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from photoholmes.utils.image import read_image, read_jpeg_data

from .base import BaseDataset


class RealisticTamperingDataset(BaseDataset):
    """
    Directory structure:
    img_dir (realistic-tampering-dataset)
    ├── Canon_60D
    │   ├── ground-truth
    |   |   └── [masks in PNG]
    │   ├── pristine
    |   |   └── [imgs in TIF]
    │   └── tampered-realistic
    |       └── [imgs in TIF]
    ├── Nikon_D90
    │   └── ...[idem above]
    ├── Nikon_D7000
    │   └── ...[idem above]
    ├── Sony_A57
    │   └── ...[idem above]
    └── readme.md
    """

    CAMERA_FOLDERS = ["Canon_60D", "Nikon_D90", "Nikon_D7000", "Sony_A57"]
    TAMP_DIR = "tampered-realistic"
    AUTH_DIR = "pristine"
    MASKS_DIR = "ground-truth"
    IMAGE_EXTENSION = ".TIF"
    MASK_EXTENSION = ".PNG"
    TAMPERED_INTENSITY_THRESHOLD = 10  # 3 level masks. Gray level is tampered.

    def _get_paths(
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = []
        mask_paths = []
        for camera_dir in self.CAMERA_FOLDERS:
            cam_im_paths, cam_mk_paths = self._get_camera_paths(
                os.path.join(img_dir, camera_dir), tampered_only
            )
            image_paths += cam_im_paths
            mask_paths += cam_mk_paths
        return image_paths, mask_paths

    def _get_camera_paths(
        self, camera_dir, tampered_only
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = glob.glob(
            os.path.join(camera_dir, self.TAMP_DIR, f"*{self.IMAGE_EXTENSION}")
        )

        mask_paths: List[Optional[str]] = [
            os.path.join(self.img_dir, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(
                os.path.join(camera_dir, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
            )

            pris_msk_paths = [None] * len(pris_paths)
            image_paths += pris_paths
            mask_paths += pris_msk_paths

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        cam_dir = image_path.split("/")[-3]
        image_filename = image_path.split("/")[-1]
        name = ".".join(image_filename.split(".")[:-1])
        mask_filename = name + self.MASK_EXTENSION
        return os.path.join(cam_dir, self.MASKS_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > self.TAMPERED_INTENSITY_THRESHOLD

    def _get_data(self, idx: int) -> Tuple[Dict, Tensor, str]:
        x = {}

        image_path = self.image_paths[idx]
        image_name = "_".join(image_path.split("/")[-2:]).split(".")[0]

        if self.image_data or self.mask_paths[idx] is None:
            image = read_image(image_path)
            if "image" in self.item_data:
                x["image"] = image
            if "original_image_size" in self.item_data or self.mask_paths[idx] is None:
                x["original_image_size"] = image.shape[-2:]
        if self.jpeg_data:
            dct, qtables = read_jpeg_data(image_path)
            if "dct_coefficients" in self.item_data:
                x["dct_coefficients"] = torch.tensor(dct)
            if "qtables" in self.item_data:
                x["qtables"] = torch.tensor(np.array(qtables))

        if self.mask_paths[idx] is None:
            mask = torch.zeros(x["original_image_size"], dtype=torch.bool)
        else:
            mask_im = read_image(self.mask_paths[idx])
            mask = self._binarize_mask(mask_im)

        return x, mask, image_name
