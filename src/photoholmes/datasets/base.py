from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from photoholmes.utils.image import read_image, read_jpeg_data


class BaseDataset(ABC, Dataset):
    def __init__(
        self,
        img_dir: str,
        item_data: List[Literal["image", "dct_coefficients", "qtables"]] = ["image"],
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

    @abstractmethod
    def _get_paths(
        self, img_dir, tampered_only
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """Abstract method that returns image and mask paths. Should make use of tampered_only attribute."""
        pass

    @abstractmethod
    def _get_mask_path(self, image_path: str) -> str:
        pass

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Dict, Tensor]:
        x, mask = self._get_data(idx)
        print(x)
        if self.transform:
            x = self.transform(**x)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return x, mask

    def _get_data(self, idx: int) -> Tuple[Dict, Tensor]:
        x = {}

        image_path = self.image_paths[idx]
        if "image" in self.item_data:
            print(image_path)
            x["image"] = read_image(image_path)
        if self.jpeg_data:
            dct, qtables = read_jpeg_data(image_path)
            if "dct_coefficients" in self.item_data:
                x["dct_coefficients"] = torch.tensor(dct)
            if "qtables" in self.item_data:
                x["qtables"] = torch.tensor(qtables)

        if self.mask_paths[idx] is None:
            arbitrary_element = list(x.values())[0]
            mask = torch.zeros_like(arbitrary_element[0, :, :])
        else:
            mask_im = read_image(self.mask_paths[idx])
            mask = self._binarize_mask(mask_im)

        return x, mask

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """Overideable method for binarizing mask images."""
        assert (mask_image <= 1).all()
        return mask_image == 1
