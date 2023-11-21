import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .abstract import AbstractDataset


class RealisticTamperingDataset(AbstractDataset):
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
        self, img_dir, tampered_only
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
        image_filenames = glob.glob(
            os.path.join(camera_dir, self.TAMP_DIR, f"*{self.IMAGE_EXTENSION}")
        )
        image_paths = [
            os.path.join(camera_dir, self.TAMP_DIR, filename)
            for filename in image_filenames
        ]
        mask_paths: List[Optional[str]] = [
            self._get_mask_path(image_path) for image_path in image_paths
        ]

        if not tampered_only:
            pris_filenames = glob.glob(
                os.path.join(camera_dir, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
            )
            pris_paths = [
                os.path.join(camera_dir, self.AUTH_DIR, filename)
                for filename in pris_filenames
            ]
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
