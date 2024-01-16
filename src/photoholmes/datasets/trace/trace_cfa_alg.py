import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from photoholmes.utils.image import read_image, read_jpeg_data

from ..base import BaseDataset


class TraceCFAAlgEndoDataset(BaseDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_endo.png, image tampered with different cfa algorithms
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_endo.png, image tampered with different cfa algorithms
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME = "original.NEF"
    FORGED_NAME = "cfa_alg_endo.png"
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

    def _get_data(self, idx: int) -> Tuple[Dict, Tensor, str]:
        x = {}

        image_path = self.image_paths[idx]
        image_name = "_".join(image_path.split("/")[-2:]).split(".")[0]

        if self.image_data or self.mask_paths[idx] is None:
            image = read_image(image_path)
            if "image" in self.item_data:
                x["image"] = image
            if "original_image_size" in self.item_data or self.mask_paths[idx] is None:
                x["original_image_size"] = image.shape[-2:]
        if self.jpeg_data:
            dct, qtables = read_jpeg_data(image_path)
            if "dct_coefficients" in self.item_data:
                x["dct_coefficients"] = torch.tensor(dct)
            if "qtables" in self.item_data:
                x["qtables"] = torch.tensor(np.array(qtables))

        if self.mask_paths[idx] is None:
            mask = torch.zeros(x["original_image_size"], dtype=torch.bool)
        else:
            mask_im = read_image(self.mask_paths[idx])
            mask = self._binarize_mask(mask_im)

        return x, mask, image_name


class TraceCFAAlgExoDataset(BaseDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_exo.png, image tampered with different cfa algorithms
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_exo.png, image tampered with different cfa algorithms
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME = "original.NEF"
    FORGED_NAME = "cfa_alg_exo.png"
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

    def _get_data(self, idx: int) -> Tuple[Dict, Tensor, str]:
        x = {}

        image_path = self.image_paths[idx]
        image_name = "_".join(image_path.split("/")[-2:]).split(".")[0]

        if self.image_data or self.mask_paths[idx] is None:
            image = read_image(image_path)
            if "image" in self.item_data:
                x["image"] = image
            if "original_image_size" in self.item_data or self.mask_paths[idx] is None:
                x["original_image_size"] = image.shape[-2:]
        if self.jpeg_data:
            dct, qtables = read_jpeg_data(image_path)
            if "dct_coefficients" in self.item_data:
                x["dct_coefficients"] = torch.tensor(dct)
            if "qtables" in self.item_data:
                x["qtables"] = torch.tensor(np.array(qtables))

        if self.mask_paths[idx] is None:
            mask = torch.zeros(x["original_image_size"], dtype=torch.bool)
        else:
            mask_im = read_image(self.mask_paths[idx])
            mask = self._binarize_mask(mask_im)

        return x, mask, image_name
