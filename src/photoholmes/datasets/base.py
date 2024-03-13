import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from photoholmes.preprocessing import PreProcessingPipeline
from photoholmes.utils.image import read_image, read_jpeg_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDataset(ABC, Dataset):
    """
    Base class for datasets.

    Subclasses must override the IMAGE_EXTENSION and MASK_EXTENSION attributes.
    The _get_paths and _get_mask_path methods must be implemented as well.
    """

    IMAGE_EXTENSION: Union[str, List[str]]
    MASK_EXTENSION: Union[str, List[str]]

    def __init__(
        self,
        dataset_path: str,
        item_data: List[Literal["image", "dct_coefficients", "qtables",]] = [
            "image",
            "dct_coefficients",
            "qtables",
        ],
        preprocessing_pipeline: Optional[PreProcessingPipeline] = None,
        only_load_tampered: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (str): Path to the dataset.
            item_data (List[Literal["image", "dct_coefficients", "qtables"]]): List of
                items to load. Possible values are "image", "dct_coefficients" and
                "qtables".
            preprocessing_pipeline (Optional[PreProcessingPipeline]): Preprocessing
                pipeline to apply to the images.
            only_load_tampered (bool): If True, only load tampered images.

        Raises:
            FileNotFoundError: If the dataset_path does not exist.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Directory {dataset_path} does not exist.")

        self.dataset_path = dataset_path
        self.only_load_tampered = only_load_tampered

        self.preprocessing_pipeline = preprocessing_pipeline
        self.item_data = (
            preprocessing_pipeline.inputs if preprocessing_pipeline else item_data
        )

        self.load_jpeg_data = (
            "dct_coefficients" in self.item_data or "qtables" in self.item_data
        )
        self.load_image_data = "image" in self.item_data

        if self.load_jpeg_data:
            self.check_jpeg_warning()
        self.check_attribute_override()

        self.image_paths, self.mask_paths = self._get_paths(
            dataset_path, only_load_tampered
        )

    @abstractmethod
    def _get_paths(
        self, dataset_path, only_load_tampered
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """Abstract method that returns image and mask paths. Should make use of
        the dataset_path and only_load_tampered arguments."""
        pass

    @abstractmethod
    def _get_mask_path(self, image_path: str) -> str:
        pass

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Dict, Tensor, str]:
        x, mask, image_name = self._get_data(idx)
        if self.preprocessing_pipeline is not None:
            x = self.preprocessing_pipeline(**x)
        return x, mask, image_name

    def _get_data(self, idx: int) -> Tuple[Dict, Tensor, str]:
        x = {}

        image_path = self.image_paths[idx]
        image_name = image_path.split("/")[-1].split(".")[0]

        if self.load_image_data:
            image = read_image(image_path)
            x["image"] = image
        if self.load_jpeg_data:
            dct, qtables = read_jpeg_data(image_path, suppress_not_jpeg_warning=True)
            if "dct_coefficients" in self.item_data:
                x["dct_coefficients"] = torch.tensor(dct)
            if "qtables" in self.item_data:
                x["qtables"] = torch.tensor(np.array(qtables))

        if self.mask_paths[idx] is None:
            mask = torch.zeros(image.shape[-2:], dtype=torch.bool)
        else:
            mask_im = read_image(self.mask_paths[idx])
            mask = self._binarize_mask(mask_im)

        return x, mask, image_name

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """Overrideable method to binarize the mask image."""
        assert (mask_image <= 1).all()
        return (mask_image == 1).float()

    def check_attribute_override(self):
        """
        Check that the subclass has overridden IMAGE_EXTENSION and MASK_EXTENSION.
        Raises an error if not.
        """
        if not hasattr(type(self), "IMAGE_EXTENSION"):
            raise NotImplementedError("Subclasses must override IMAGE_EXTENSION")
        if not hasattr(type(self), "MASK_EXTENSION"):
            raise NotImplementedError("Subclasses must override MASK_EXTENSION")

    def check_jpeg_warning(self):
        """
        Check if the images are in JPEG format. If not, a warning is issued.
        """
        if not isinstance(self.IMAGE_EXTENSION, list):
            image_ext = [self.IMAGE_EXTENSION]
        else:
            image_ext = self.IMAGE_EXTENSION
        if not all(
            [
                ext in [".jpg", ".jpeg", ".JPG", ".JPEG", "jpg", "jpeg", "JPEG", "JPG"]
                for ext in image_ext
            ]
        ):
            logger.warning(
                "Not all images are in JPEG format. When needed, an approximation will "
                "be loaded by compressing the image in quality 100."
            )
