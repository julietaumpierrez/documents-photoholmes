import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .abstract import AbstractDataset  # type: ignore


class ColumbiaDataset(AbstractDataset):
    """
    Directory structure:
    img_dir (Columbia Uncompressed Image Splicing Detection)
    ├── 4cam_auth
    │   ├── [images in TIF]
    |   └── edgemask
    |       └── [masks in JPG]
    ├── 4cam_splc
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
    ) -> Tuple[List[str], List[float | str] | List[str]]:
        image_filenames = os.listdir(os.path.join(img_dir, self.TAMP_DIR))
        image_paths = [
            os.path.join(img_dir, self.TAMP_DIR, filename)
            for filename in image_filenames
            if super(ColumbiaDataset, ColumbiaDataset).file_extension(filename)
            == self.IMAGE_EXTENSION
        ]
        mask_paths = [self._get_mask_path(image_path) for image_path in image_paths]
        if not tampered_only:
            pris_paths = os.listdir(os.path.join(img_dir, self.TAMP_DIR))
            pris_msk_paths = [
                np.NaN for i in range(len(pris_paths))
            ]  # NaN for pristine image flag
            image_paths += pris_paths
            mask_paths += pris_msk_paths
        return image_paths, mask_paths

    def _get_mask_path(self, image_path) -> str:
        image_filename = image_path.split("/")[-1]
        image_name_list = ".".join(image_filename.split(".")[:-1]).split("_")
        mask_name = "_".join(image_name_list + ["edgemask"])
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.MASKS_DIR, mask_filename)

    def _get_data(self, idx) -> Tuple[Tensor, Tensor]:
        image_path = os.path.join(self.img_dir, self.image_paths[idx])
        image = self._read_image(image_path)
        mask_path = os.path.join(self.img_dir, self.mask_paths[idx])
        if mask_path == np.NaN:
            mask = torch.zeros_like(image)
        else:
            mask = self._read_image(mask_path)
            mask = self._binarize_mask(mask)
        return image, mask

    def _binarize_mask(self, mask_image) -> Tensor:
        return mask_image[:, :, self.TAMPERED_COLOR_INDEX] > 0
