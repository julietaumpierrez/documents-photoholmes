import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class AutospliceDataset(BaseDataset):
    """
    Directory structure:
    img_dir (AutoSplice)
    ├── Authentic
    │   └── [Authentic images in jpg]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Forged_JPEG90
    │   └── [Forged images in jpg compressed with Q=90]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Mask
    │   └── [Mask in png]
    └── Possibly more files
    """

    FORGED_DIR: str
    AUTH_DIR: str = "Authentic"
    MASK_DIR: str = "Mask"
    IMAGE_EXTENSION: str = ".jpg"
    MASK_EXTENSION: str = ".png"

    def _get_paths(
        self, dataset_path: str, only_load_tampered: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = glob.glob(
            os.path.join(dataset_path, self.FORGED_DIR, f"*{self.IMAGE_EXTENSION}")
        )
        mask_paths: List[Optional[str]] = [
            os.path.join(dataset_path, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not only_load_tampered:
            pris_paths = glob.glob(
                os.path.join(dataset_path, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
            )

            pris_msk_paths = [None] * len(pris_paths)
            image_paths += pris_paths
            mask_paths += pris_msk_paths

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        image_filename = image_path.split("/")[-1]
        image_name_list = image_filename.split(".")[0].split("_")[0]
        mask_name = image_name_list + "_mask"
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.MASK_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > 0


class Autosplice100Dataset(AutospliceDataset):
    """
    Directory structure:
    img_dir (AutoSplice)
    ├── Authentic
    │   └── [Authentic images in jpg]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Forged_JPEG90
    │   └── [Forged images in jpg compressed with Q=90]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Mask
    │   └── [Mask in png]
    └── Possibly more files
    """

    FORGED_DIR = "Forged_JPEG100"


class Autosplice90Dataset(AutospliceDataset):
    """
    Directory structure:
    img_dir (AutoSplice)
    ├── Authentic
    │   └── [Authentic images in jpg]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Forged_JPEG90
    │   └── [Forged images in jpg compressed with Q=90]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Mask
    │   └── [Mask in png]
    └── Possibly more files
    """

    FORGED_DIR = "Forged_JPEG90"


class Autosplice75Dataset(AutospliceDataset):
    """
    Directory structure:
    img_dir (AutoSplice)
    ├── Authentic
    │   └── [Authentic images in jpg]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Forged_JPEG90
    │   └── [Forged images in jpg compressed with Q=90]
    ├── Forged_JPEG100
    │   └── [Forged images in jpg compressed with Q=100]
    ├── Mask
    │   └── [Mask in png]
    └── Possibly more files
    """

    FORGED_DIR = "Forged_JPEG75"
