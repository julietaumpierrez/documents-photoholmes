import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset


class AbstractDataset(ABC, Dataset):
    def __init__(
        self, img_dir, transform=None, mask_transform=None, tampered_only=False
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.tampered_only = tampered_only
        self.image_paths, self.mask_paths = self._get_paths(img_dir, tampered_only)

    @abstractmethod
    def _get_paths(self, img_dir, tampered_only) -> Tuple[List[str], List[str]]:
        """Abstract method that returns image and mask paths. Should make use of tampered_only attribute."""
        pass

    @abstractmethod
    def _get_mask_path(self, image_path) -> str:
        pass

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[Dict, Tensor]:
        x, mask = self._get_data(idx)
        if self.transform:
            image = self.transform(**x)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return x, mask

    def _get_data(self, idx) -> Tuple[Dict, Tensor]:
        image_path = os.path.join(self.img_dir, self.image_paths[idx])
        image = self._read_image(image_path)
        mask_path = os.path.join(self.img_dir, self.mask_paths[idx])
        mask_im = self._read_image(mask_path)
        mask = self._binarize_mask(mask_im)
        x = {"image": image}
        return x, mask

    @staticmethod
    def _read_image(path):
        return torch.from_numpy(np.asarray(Image.open(path)))

    def _binarize_mask(self, mask_image) -> Tensor:
        """Overideable method for binarizing mask images."""
        assert (mask_image <= 1).all()
        return mask_image == 1
