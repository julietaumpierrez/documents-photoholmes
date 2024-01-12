from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from photoholmes.utils.image import read_image, read_jpeg_data


class BaseDataset(ABC, Dataset):
    def __init__(
        self,
        img_dir: str,
        item_data: List[
            Literal[
                "image",
                "dct_coefficients",
                "qtables",
                "original_image_size",
            ]
        ] = [
            "image",
            "dct_coefficients",
            "qtables",
            "original_image_size",
        ],
        # TODO add typing for transforms
        transform=None,
        mask_transform=None,
        tampered_only: bool = False,
    ):
        self.img_dir = img_dir
        self.item_data = item_data
        self.transform = transform
        self.mask_transform = mask_transform
        self.tampered_only = tampered_only
        self.image_paths, self.mask_paths = self._get_paths(img_dir, tampered_only)
        self.jpeg_data = (
            "dct_coefficients" in self.item_data or "qtables" in self.item_data
        )
        self.image_data = (
            "image" in self.item_data or "original_image_size" in self.item_data
        )

    @abstractmethod
    def _get_paths(
        self, img_dir, tampered_only
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """Abstract method that returns image and mask paths. Should make use of
        tampered_only attribute."""
        pass

    @abstractmethod
    def _get_mask_path(self, image_path: str) -> str:
        pass

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Dict, Tensor, str]:
        x, mask, image_name = self._get_data(idx)
        if self.transform:
            x = self.transform(**x)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return x, mask, image_name

    def _get_data(self, idx: int) -> Tuple[Dict, Tensor, str]:
        x = {}

        image_path = self.image_paths[idx]
        image_name = image_path.split("/")[-1].split(".")[0]

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

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """Overideable method for binarizing mask images."""
        assert (mask_image <= 1).all()
        return (mask_image == 1).float()
