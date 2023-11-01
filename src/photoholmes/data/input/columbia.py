import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .abstract import AbstractDataset


class ColumbiaDataset(AbstractDataset):
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
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[Optional[str]]]:
        image_filenames = glob.glob(
            os.path.join(img_dir, self.TAMP_DIR, f"*{self.IMAGE_EXTENSION}")
        )
        image_paths = [
            os.path.join(img_dir, self.TAMP_DIR, filename)
            for filename in image_filenames
        ]
        mask_paths: List[Optional[str]] = [
            self._get_mask_path(image_path) for image_path in image_paths
        ]

        if not tampered_only:
            pris_filenames = glob.glob(
                os.path.join(img_dir, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
            )
            pris_paths = [
                os.path.join(img_dir, self.AUTH_DIR, filename)
                for filename in pris_filenames
            ]
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
        return mask_image[:, :, self.TAMPERED_COLOR_INDEX] > 0
