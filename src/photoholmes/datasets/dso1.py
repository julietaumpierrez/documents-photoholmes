import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class DSO1Dataset(BaseDataset):
    """
    Directory structure:
    img_dir (tifs-database)
    ├── DSO-1
    │   ├── [images in png, normal for untampered, splicing for forged]
    ├── DSO-1-Fake-Images-Masks
    │   ├── [masks in png]
    └── Possibly more folders
    """

    TAMP_DIR = "DSO-1"
    AUTH_DIR = "DSO-1"
    MASKS_DIR = "DSO-1-Fake-Images-Masks"
    IMAGE_EXTENSION = ".png"
    MASK_EXTENSION = ".png"

    IMAGE_DIR = "DSO-1"
    MASKS_DIR = "DSO-1-Fake-Images-Masks"
    IMAGE_EXTENSION = ".png"
    MASK_EXTENSION = ".png"
    TAMPERED_TAG = "splicing"
    UNTAMPERED_TAG = "normal"
    MASK_TAG = "splicing"

    def _get_paths(
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        tag = self.TAMPERED_TAG if tampered_only else ""
        image_paths = glob.glob(
            os.path.join(img_dir, self.IMAGE_DIR, f"{tag}*{self.IMAGE_EXTENSION}")
        )
        mask_paths: List[Optional[str]] = [
            self._get_mask_path(image_path) if self._is_tampered(image_path) else None
            for image_path in image_paths
        ]
        return image_paths, mask_paths

    def _is_tampered(self, image_path: str) -> bool:
        filename = os.path.basename(image_path)
        tag = filename.split("-")[0]
        return tag == self.TAMPERED_TAG

    def _get_mask_path(self, image_path: str) -> str:
        img_dir = "/".join(image_path.split("/")[:-2])
        im_filename = image_path.split("/")[-1]
        mask_path = os.path.join(img_dir, self.MASKS_DIR, im_filename)
        return mask_path

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] == 0


class DSO1OSNDataset(DSO1Dataset):
    """
    Directory structure:
    img_dir (tifs-database)
    ├── DSO-1
    │   ├── [images in png, normal for untampered, splicing for forged]
    ├── DSO_Whatsapp
    │   ├── [Images in jpeg]
    |   ├── DSO_GT
    |   |   └── [masks in png]
    └── Possibly more folders
    """

    TAMP_DIR = "DSO_Whatsapp"
    IMAGE_EXTENSION = ".jpeg"
    IMAGE_DIR = "DSO_Whatsapp"
    MASK_EXTENSION = "_gt.png"
    MASKS_DIR = "DSO_WHATSAPP/DSO_GT"

    def _get_mask_path(self, image_path: str) -> str:
        img_dir = "/".join(image_path.split("/")[:-2])
        im_filename = image_path.split("/")[-1]
        im_filename = im_filename.replace(self.IMAGE_EXTENSION, self.MASK_EXTENSION)
        mask_path = os.path.join(img_dir, self.MASKS_DIR, im_filename)
        return mask_path

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > 0
