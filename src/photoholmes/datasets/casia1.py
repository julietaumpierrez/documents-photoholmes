import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class Casia1SplicingDataset(BaseDataset):
    """
    Directory structure:
    img_dir (CASIA 1.0 dataset)
    ├── Au
    │   └── [Authentic images in jpg]
    ├── Tp
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    |   ├── Sp
    |       └── [Spliced images in jpg]
    ├── CASIA 1.0 groundtruth
    │   ├── CM
    │   │   └── [Copy Move masks in png]
    |   ├── Sp
    |       └── [Spliced masks in png]
    └── Possibly more files
    """

    SP_DIR = "Tp/Sp"
    AUTH_DIR = "Au"
    SP_MASKS_DIR = "CASIA 1.0 groundtruth/Sp"
    IMAGE_EXTENSION = ".jpg"
    MASK_EXTENSION = ".png"

    def _get_paths(
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = glob.glob(
            os.path.join(img_dir, self.SP_DIR, f"*{self.IMAGE_EXTENSION}")
        )
        mask_paths: List[Optional[str]] = [
            os.path.join(self.img_dir, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(
                os.path.join(img_dir, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
            )

            pris_msk_paths = [None] * len(pris_paths)
            image_paths += pris_paths
            mask_paths += pris_msk_paths

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        image_filename = image_path.split("/")[-1]
        image_name_list = ".".join(image_filename.split(".")[:-1]).split("_")
        mask_name = "_".join(image_name_list + ["gt"])
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.SP_MASKS_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > 0


class Casia1SplicingOSNDataset(Casia1SplicingDataset):
    """
    Directory structure:
    img_dir (CASIA 1.0 dataset)
    ├── Au
    │   └── [Authentic images in jpg]
    ├── Casia_Facebook
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    |   ├── Sp
    |   |   └── [Spliced images in jpg]
    |   ├── CASIA_GT
    |   |   └── [Groundtruth masks in png]
    ├── CASIA 1.0 groundtruth
    │   ├── CM
    │   │   └── [Copy Move masks in png]
    |   ├── Sp
    |   |   └── [Spliced masks in png]
    └── Possibly more files
    """

    SP_DIR = "Casia_Facebook/Sp"
    IMAGE_EXTENSION = ".jpg"
    SP_MASKS_DIR = "Casia_Facebook/CASIA_GT"


class Casia1CopyMoveDataset(BaseDataset):
    """
    Directory structure:
    img_dir (CASIA 1.0 dataset)
    ├── Au
    │   └── [Authentic images in jpg]
    ├── Tp
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    |   ├── Sp
    |   |   └── [Spliced images in jpg]
    ├── CASIA 1.0 groundtruth
    │   ├── CM
    │   │   └── [Copy Move masks in png]
    |   ├── Sp
    |   |   └── [Spliced masks in png]
    └── Possibly more files
    """

    CM_DIR = "Tp/CM"
    AUTH_DIR = "Au"
    CM_MASKS_DIR = "CASIA 1.0 groundtruth/CM"
    IMAGE_EXTENSION = ".jpg"
    MASK_EXTENSION = ".png"

    def _get_paths(
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = glob.glob(
            os.path.join(img_dir, self.CM_DIR, f"*{self.IMAGE_EXTENSION}")
        )
        mask_paths: List[Optional[str]] = [
            os.path.join(self.img_dir, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(
                os.path.join(img_dir, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
            )

            pris_msk_paths = [None] * len(pris_paths)
            image_paths += pris_paths
            mask_paths += pris_msk_paths

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        image_filename = image_path.split("/")[-1]
        image_name_list = ".".join(image_filename.split(".")[:-1]).split("_")
        mask_name = "_".join(image_name_list + ["gt"])
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.CM_MASKS_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > 0


class Casia1CopyMoveOSNDataset(Casia1CopyMoveDataset):
    """
    Directory structure:
    img_dir (CASIA 1.0 dataset)
    ├── Au
    │   └── [Authentic images in jpg]
    ├── Casia_Facebook
    │   ├── CM
    │   │   └── [Copy Move images in jpeg]
    |   ├── Sp
    |   |   └── [Spliced images in jpeg]
    |   ├── CASIA_GT
    |   |   └── [Groundtruth masks in png]
    ├── CASIA 1.0 groundtruth
    │   ├── CM
    │   │   └── [Copy Move masks in png]
    |   ├── Sp
    |   |   └── [Spliced masks in png]
    └── Possibly more files
    """

    CM_DIR = "Casia_Facebook/CM"
    IMAGE_EXTENSION = ".jpg"
    CM_MASKS_DIR = "Casia_Facebook/CASIA_GT"
