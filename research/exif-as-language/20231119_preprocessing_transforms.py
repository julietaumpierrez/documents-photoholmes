# %%
import os

from cv2 import norm

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")


# %%
from typing import List, TypeVar, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.utils.preprocessing.base import PreprocessingTransform

T = TypeVar("T", Tensor, NDArray)


class Normalize_PH(PreprocessingTransform):
    """
    Normalize an image.

    Args:
        image: Image to be normalized.
        **kwargs: Additional keyword arguments to passthrough.
    Returns:
        A dictionary with the following key-value pairs:
            - "image": The normalized image.
            - **kwargs: The additional keyword arguments passed through unchanged.
    """

    def __init__(self, mean: Union[List[int], T], std: Union[List[int], T]) -> None:
        if isinstance(mean, list):
            self.mean = np.array(mean)
        else:
            self.mean = mean
        if isinstance(std, list):
            self.std = np.array(std)
        else:
            self.std = std

    def __call__(self, image: T, **kwargs):
        if isinstance(image, Tensor):
            mean = torch.as_tensor(self.mean)
            std = torch.as_tensor(self.std)

            image = (image.float() - mean.view(3, 1, 1)) / std.view(3, 1, 1)

        else:
            mean = self.mean
            std = self.std

            image = (image.astype(np.float32) - mean.reshape((1, 1, -1))) / std.reshape(
                (1, 1, -1)
            )

        return {"image": image, **kwargs}


# %%
import time

# %%
normalize = Normalize_PH(mean=[1, 1, 1], std=[1, 1, 1])
image = torch.normal(mean=1, std=1, size=(3, 200, 200))

t0 = time.time_ns()
norm_image = normalize(image=image)["image"]
tf = time.time_ns()
print(f"PH: {tf - t0} ns")
# %%
from torchvision.transforms import Normalize as Normalize_torch

normalize_torch = Normalize_torch(mean=[1, 1, 1], std=[1, 1, 1])

t0 = time.time_ns()
norm_image_torch = normalize_torch(image)
tf = time.time_ns()
print(f"Torch: {tf - t0} ns")

# %%
image_numpy = image.permute(1, 2, 0).numpy()
norm_image_numpy = normalize(image_numpy)["image"]
# %%
norm_image_torch == torch.from_numpy(norm_image_numpy).permute(2, 0, 1)


import matplotlib.pyplot as plt

# %%
from PIL import Image


class GrayToRGB(PreprocessingTransform):
    """
    Converts an grayscale image to RGB
    """

    def __call__(self, image: T, **kwargs):
        if isinstance(image, Tensor):
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif isinstance(image, np.ndarray):
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        return {"image": image, **kwargs}


# %%
image = np.array(Image.open("data/img00.png").convert("L"))

gray_to_rgb = GrayToRGB()
plt.imshow(image, cmap="gray")
# %%
image_rgb = gray_to_rgb(image=image)["image"]
plt.imshow(image_rgb, cmap="gray")

# %%
image_tensor = torch.as_tensor(image)
# %%
image_tensor_rgb = gray_to_rgb(image=image_tensor)["image"]


# %%
def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB if necessary
    Params: Input image
    Return: RGB image"""
    return image.convert("RGB")


image_pil = Image.open("data/img00.png").convert("L")
image_rgb_pil = np.array(_convert_image_to_rgb(image_pil))

plt.imshow(image_rgb_pil, cmap="gray")
# %%
(image_rgb_pil == image_rgb).all()

# %%
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

from photoholmes.models.exif_as_language import EXIF_SC


def _transform(mean: tuple, std: tuple) -> Compose:
    """Compose transforms
    Params:
        mean(tuple): mean values for normalization
        std(tuple): std values for normalization
    Return: Composed transforms"""
    return Compose(
        [
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ]
    )


def preprocess(
    image: torch.Tensor,
    mean: tuple = (0.48145466, 0.4578275, 0.40821073),
    std: tuple = (0.26862954, 0.26130258, 0.27577711),
) -> torch.Tensor:
    """Preprocess image with _transform
    Params: Input image
            mean (tuple): mean values for normalization
            std (tuple): std values for normalization
    Return: Preprocessed image"""
    toPIL = ToPILImage()
    image = toPIL(image)
    func = _transform(mean, std)
    return func(image)


method = EXIF_SC(
    "distilbert",
    "resnet50",
    device="cpu",
    state_dict_path="weights/exif/pruned_weights.pth",
)

# %%
from torchvision.io import read_image

from photoholmes.models.exif_as_language.preprocessing import exif_preprocessing

image = read_image("data/img00.png") / 255
image = image.to(dtype=torch.float32)

# %%
image_input_old = preprocess(image=image)
# %%
out = method.predict(img=image_input_old)

plt.imshow(out["ms"])
plt.show()
# %%
image_input = exif_preprocessing(image=image)
# %%
out = method.predict(img=image_input["image"])

plt.imshow(out["ms"])
plt.show()

# %%
