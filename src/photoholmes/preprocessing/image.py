from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import cv2 as cv
import numpy as np
import torch
from numpy.typing import NDArray
from PIL.Image import Image
from torch import Tensor

from photoholmes.preprocessing.base import BasePreprocessing

T = TypeVar("T", Tensor, NDArray)


class ZeroOneRange(BasePreprocessing):
    """
    Changes the image range from [0, 255] to [0, 1].
    """

    def __call__(self, image: T, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (T): Image to be normalized.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The normalized image.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        if image.dtype == np.uint8 or image.dtype == torch.uint8:
            image = image / 255
        elif image.max() > 1:
            image = image / 255
        return {"image": image, **kwargs}


class Normalize(BasePreprocessing):
    """
    Normalize an image. When called with an image with the mean and std of the class
    instance, it returns an image with mean 0 and std of 1.
    """

    def __init__(
        self,
        mean: Union[Tuple[float, ...], T],
        std: Union[Tuple[float, ...], T],
    ) -> None:
        """
        Args:
            mean (Union[Tuple[float, ...], T]): Mean value for each channel.
            std (Union[Tuple[float, ...], T]): Standard deviation for each channel.
        """
        if isinstance(mean, tuple):
            self.mean = np.array(mean)
        else:
            self.mean = mean
        if isinstance(std, tuple):
            self.std = np.array(std)
        else:
            self.std = std

    def __call__(self, image: T, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (T): Image to be normalized.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The normalized image.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        if isinstance(image, Tensor):
            mean = torch.as_tensor(self.mean, dtype=torch.float32)
            std = torch.as_tensor(self.std, dtype=torch.float32)

            if image.ndim == 3:
                mean = mean.view(3, 1, 1)
                std = std.view(3, 1, 1)

            t_image = (image.float() - mean) / std

        else:
            mean = self.mean
            std = self.std
            if image.ndim == 3:
                mean = mean.reshape((1, 1, -1))
                std = std.reshape((1, 1, -1))

            t_image = (image.astype(np.float32) - mean) / std

        return {"image": t_image, **kwargs}


class ToTensor(BasePreprocessing):
    """
    Converts a numpy array to a PyTorch tensor.
    """

    def __call__(self, image: NDArray, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (NDArray): Image to be converted to a tensor.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The input image as a PyTorch tensor.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        if isinstance(image, Image):
            t_image = torch.from_numpy(image)
            if t_image.ndim == 3:
                t_image = t_image.permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            t_image = torch.from_numpy(image)
            if t_image.ndim == 3:
                t_image = t_image.permute(2, 0, 1)
        elif isinstance(image, Tensor):
            t_image = image
        else:
            raise ValueError(f"image type {type(image)} isn't handled by ToTensor")

        for k in kwargs:
            if isinstance(kwargs[k], Tensor):
                continue
            elif isinstance(kwargs[k], list):
                kwargs[k] = np.array(kwargs[k])
            kwargs[k] = torch.from_numpy(kwargs[k])

        return {"image": t_image, **kwargs}


class ToNumpy(BasePreprocessing):
    """
    Converts inputs to numpy arrays. If input is already a numpy array,
    it leaves it as is.
    """

    def __call__(
        self, image: Optional[Union[T, Image]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Args:
            image(Optional[Union[T, Image]]): Image to be converted to a tensor.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The input image as a numpy array.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        t_image = None
        if isinstance(image, Tensor):
            t_image = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, np.ndarray):
            t_image = image.copy()
        elif image is not None:
            t_image = np.array(image)

        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                continue
            elif isinstance(v, Tensor):
                kwargs[k] = v.cpu().numpy()
            else:
                kwargs[k] = v

        if t_image is None:
            return {**kwargs}
        else:
            return {"image": t_image, **kwargs}


class RGBtoGray(BasePreprocessing):
    """
    Converts an RGB image to grayscale, following the ITU-R BT.601 stardard.
    """

    def __call__(self, image: T, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (T): Image to be converted to grayscale.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]:A dictionary with the following key-value pairs:
                - "image": The input image as a grayscale numpy array or PyTorch tensor.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        if isinstance(image, Tensor):
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = image.unsqueeze(0)
        else:
            image = (
                0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            )
            image = image[..., np.newaxis]
        return {"image": image, **kwargs}


class RoundToUInt(BasePreprocessing):
    """
    Rounds the input float tensor and converts it to an unsigned integer.
    """

    def __call__(self, image: Tensor, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (Tensor): Image to be converted to rounded into uint.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The input image rounded as a PyTorch tensor.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        rounded_image = torch.round(image).byte()
        return {"image": rounded_image, **kwargs}


class GrayToRGB(BasePreprocessing):
    """
    Converts an grayscale image to RGB, done by repeating the image along the three channels.
    """

    def __call__(self, image: T, **kwargs):
        """
        Args:
            image (T): Image to be converted to grayscale.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The input image as a grayscale numpy array or PyTorch tensor.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        if isinstance(image, Tensor):
            if image.ndim == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            elif image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
        return {"image": image, **kwargs}


class GetImageSize(BasePreprocessing):
    """
    Get the size of the image.
    """

    def __call__(self, image: T, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (T): Image to be converted to grayscale.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The input image as a grayscale numpy array or PyTorch tensor.
                - "image_size": The size of the input image.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        if isinstance(image, Tensor):
            size = tuple(image.shape[1:])
        elif isinstance(image, np.ndarray):
            size = image.shape[:2]
        elif isinstance(image, Image):
            size = image.size
        else:
            raise ValueError(f"Image type not supported: {type(image)}")
        return {"image": image, "image_size": size, **kwargs}


class RGBtoYCrCb(BasePreprocessing):
    """
    Converts an RGB image to YCrCb.
    """

    def __call__(self, image: T, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image (T): Image to be converted to YCrCb.
            **kwargs: Additional keyword arguments to passthrough.

        Returns:
            Dict[str, Any]: A dictionary with the following key-value pairs:
                - "image": The input image as a YCrCb PyTorch tensor.
                - **kwargs: The additional keyword arguments passed through unchanged.
        """
        np_image = ToNumpy()(image)["image"]
        t_np_image = cv.cvtColor(np_image, cv.COLOR_RGB2YCrCb)
        t_image = ToTensor()(t_np_image)["image"]

        return {"image": t_image, **kwargs}
