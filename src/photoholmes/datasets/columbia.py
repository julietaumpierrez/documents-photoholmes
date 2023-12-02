import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class ColumbiaDataset(BaseDataset):
    """
    Directory structure:
    img_dir (Columbia Uncompressed Image Splicing Detection)
    ├── 4cam_auth
    │   ├── [images in TIF]
    ├── 4cam_splc
    │   ├── [images in TIF]
    |   └── edgemask
    |       └── [masks in JPG]
    └── README.txt
    """

    TAMP_DIR = "4cam_splc"
    AUTH_DIR = "4cam_auth"
    MASKS_DIR = "4cam_splc/edgemask"
    IMAGE_EXTENSION = ".tif"
    MASK_EXTENSION = ".jpg"
    TAMPERED_COLOR_INDEX = 1  # Green

    def _get_paths(
        self, img_dir, tampered_only
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = glob.glob(
            os.path.join(img_dir, self.TAMP_DIR, f"*{self.IMAGE_EXTENSION}")
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
        mask_name = "_".join(image_name_list + ["edgemask"])
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.MASKS_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[self.TAMPERED_COLOR_INDEX, :, :] > 0
