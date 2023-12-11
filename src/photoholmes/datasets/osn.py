import glob
import os
from typing import List, Tuple

from torch import Tensor

from .base import BaseDataset


class OSNDataset(BaseDataset):
    """
    Directory structure:
    img_dir (ImageForgeriesOSN_Dataset)
    ├── [base directory]
    │   └── ...[img in PNG/JPG/TIF]
    ├── [base_directory + social_network_modification]
    │   └── ...[img in PNG/JPG/TIF]
    ├── [base_directory_masks]
        └── ...[masks in PNG/JPG/TIF]
    """

    BASE_FOLDERS = ["CASIA", "Columbia", "DSO"]
    MASKS_DIR_TAG = "_GT"
    MASKS_FILENAME_TAG = "_gt"
    IMAGE_EXTENSIONS = [".tif", ".jpg", ".jpeg", ".png"]
    MASK_EXTENSION = ".png"
    TAMPERED_INTENSITY_THRESHOLD = 125

    def _get_paths(
        self, img_dir: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        image_paths = []
        mask_paths = []
        for base_dir in self.BASE_FOLDERS:
            for folder in glob.glob(os.path.join(img_dir, base_dir + "*")):
                if folder != os.path.join(img_dir, base_dir) + self.MASKS_DIR_TAG:
                    folder_im_paths = [
                        os.path.join(folder, filename)
                        for filename in os.listdir(folder)
                    ]
                    image_paths += folder_im_paths
        mask_paths = [self._get_mask_path(image_path) for image_path in image_paths]
        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        img_path_list = image_path.split("/")
        img_filename = img_path_list[-1]
        img_folder = img_path_list[-2]
        img_dir = "/".join(img_path_list[:-2])
        name = ".".join(img_filename.split(".")[:-1]) + self.MASKS_FILENAME_TAG
        mask_filename = name + self.MASK_EXTENSION
        mask_folder = img_folder.split("_")[0] + self.MASKS_DIR_TAG
        return os.path.join(img_dir, mask_folder, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        return mask_image[0, :, :] > self.TAMPERED_INTENSITY_THRESHOLD
