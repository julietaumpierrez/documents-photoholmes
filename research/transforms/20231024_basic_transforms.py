# %%
import os

from PIL import Image
from sympy import im

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from typing import Dict, TypeVar

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.utils.preprocessing.base import BaseTransform, PreProcessingPipeline

T = TypeVar("T", Tensor, NDArray)


class RGBtoGray(BaseTransform):
    def __call__(self, image: T, **kwargs) -> Dict[str, T]:
        if isinstance(image, Tensor):
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            image = (
                0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            )
        return {"image": image, **kwargs}


class ToTensor(BaseTransform):
    def __call__(self, image: NDArray, **kwargs) -> Dict[str, T]:
        t_image = torch.from_numpy(image)
        if t_image.ndim == 3:
            t_image = t_image.permute(2, 0, 1)
        return {"image": t_image, **kwargs}


class Normalize(BaseTransform):
    def __call__(self, image: T, **kwargs) -> Dict[str, T]:
        if image.dtype == np.uint8 or image.dtype == torch.uint8:
            image = image / 255
        elif image.max() > 1:
            image = image / 255
        return {"image": image, **kwargs}


# %%
image = np.array(Image.open("data/img00.png"))
t_image = torch.from_numpy(image).permute(2, 0, 1)
plt.imshow(image)
image.shape

# %%
normalize = Normalize()
transform = RGBtoGray()
to_tensor = ToTensor()

# %%
image_T = normalize(image=image, mask=np.zeros(image.shape))
image_T
# %%
t_image_T = normalize(image=t_image, mask=np.zeros(t_image.shape))
t_image_T
# %%
image_T = transform(image, mask=np.zeros(image.shape))
plt.imshow(image_T["image"], cmap="gray")
image_T["image"].shape
# %%
image_T = normalize(image, mask=np.zeros(image.shape))
image_T = to_tensor(**image_T)
image_T = transform(**image_T)
plt.imshow(image_T["image"], cmap="gray")
image_T["image"].shape
# %%
t_image_T = normalize(t_image, mask=np.zeros(image.shape))
t_image_T = to_tensor(**t_image_T)
t_image_T = transform(**t_image_T)
plt.imshow(t_image_T["image"], cmap="gray")
image_T["image"].shape

# %%
pipeline = PreProcessingPipeline([Normalize(), RGBtoGray(), ToTensor()])

# %%
pipeline(image=image, mask=np.zeros(image.shape))
# %%
