import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .abstract import AbstractDataset


class CoverageDataset(AbstractDataset):
    """
    Directory structure:
    img_dir (COVERAGE)
    ├── image
    │   ├── pristine images as {NUM}.tif
    │   └── tampered images as {NUM}t.tif
    ├── label
    |   |   (tags for different types of processing)
    │   └── ...[tags as .mat files]
    ├── mask
    |   ├── duplicated region {NUM}copy.tif
    |   ├── SGO region {NUM}paste.tif
    │   └── forged region {NUM}forged.tif
    └── README.md
    """

    IMAGE_DIR = "image"
    MASKS_DIR = "mask"
    IMAGE_EXTENSION = ".tif"
    MASK_EXTENSION = ".tif"
    TAMPERED_TAG = "t"
    MASK_TAG = "forged"

    TAMPERED_INTENSITY_THRESHOLD = 125

    def _get_paths(
        self, img_dir, tampered_only
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        tag = self.TAMPERED_TAG if tampered_only else ""
        image_paths = glob.glob(
            os.path.join(img_dir, self.IMAGE_DIR, f"*{tag}{self.IMAGE_EXTENSION}")
        )
        mask_paths: List[Optional[str]] = [
            self._get_mask_path(image_path) if self._is_tampered(image_path) else None
            for image_path in image_paths
        ]
        return image_paths, mask_paths

    def _is_tampered(self, image_path: str) -> bool:
        tag = image_path[image_path.rindex(".") - 1]
        return tag == self.TAMPERED_TAG

    def _get_mask_path(self, image_path: str) -> str:
        img_dir = "/".join(image_path.split("/")[:-2])
        im_filename = image_path.split("/")[-1]
        im_index = im_filename[: im_filename.rindex(".") - 1]
        mask_path = os.path.join(
            img_dir, self.MASKS_DIR, im_index + self.MASK_TAG + self.MASK_EXTENSION
        )
        return mask_path

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > self.TAMPERED_INTENSITY_THRESHOLD
