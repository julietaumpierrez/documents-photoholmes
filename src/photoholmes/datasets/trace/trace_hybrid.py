import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from ..base import BaseDataset


class TraceHybridExoDataset(BaseDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_exo.png, image tampered with a combination of different
    |       pipelines using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_exo.png, image tampered with a combination of different
    |       pipelines using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME = "original.NEF"
    FORGED_NAME = "hybrid_select_exo.png"
    MASK_NAME = "mask_exo.png"

    def _get_paths(
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = glob.glob(os.path.join(img_dir, "*", self.FORGED_NAME))
        mask_paths: List[Optional[str]] = [
            os.path.join(self.img_dir, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(os.path.join(img_dir, "*", self.AUTH_NAME))

            pris_msk_paths = [None] * len(pris_paths)
            image_paths += pris_paths
            mask_paths += pris_msk_paths

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        image_dir = os.path.dirname(image_path)
        return os.path.join(image_dir, self.MASK_NAME)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > 0


class TraceHybridEndoDataset(BaseDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_endo.png, image tampered with a combination of different
    |       pipelines using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_endo.png, image tampered with a combination of different
    |       pipelines using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME = "original.NEF"
    FORGED_NAME = "hybrid_select_endo.png"
    MASK_NAME = "mask_endo.png"

    def _get_paths(
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = glob.glob(os.path.join(img_dir, "*", self.FORGED_NAME))
        mask_paths: List[Optional[str]] = [
            os.path.join(self.img_dir, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(os.path.join(img_dir, "*", self.AUTH_NAME))

            pris_msk_paths = [None] * len(pris_paths)
            image_paths += pris_paths
            mask_paths += pris_msk_paths

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        image_dir = os.path.dirname(image_path)
        return os.path.join(image_dir, self.MASK_NAME)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > 0
