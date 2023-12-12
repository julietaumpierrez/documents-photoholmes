# %%
import os
from typing import Tuple

from torch import Tensor

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
import numpy as np

original_image_size = (512, 512)

output_size1 = (500, 500)
output_size2 = (520, 520)
output_size3 = (512, 512)
output_size4 = (500, 520)
output_size5 = (520, 500)
# %%
import torch
import torch.nn.functional as F


def resize_tensor(tensor, originl_image_size):
    # Add batch and channel dimensions (BxCxHxW)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    print(tensor.shape)

    # Resize using interpolate
    resized_tensor = F.interpolate(
        tensor, size=originl_image_size, mode="bilinear", align_corners=True
    )
    print(resized_tensor.shape)
    # Remove added dimensions
    resized_tensor = resized_tensor.squeeze()
    print(resized_tensor.shape)
    return resized_tensor


# %%
# Original and output sizes
original_size = (512, 512)
output_sizes = [(500, 500), (520, 520), (512, 512), (500, 520), (520, 500)]

# Create and resize tensors
resized_tensors = []
for size in output_sizes:
    # Create a tensor of 'ones' for the given size
    tensor = torch.ones(size, dtype=torch.float32)

    # Resize the tensor to the original size
    resized_tensor = resize_tensor(tensor, original_size)
    resized_tensors.append(resized_tensor)
# %%
# Example: Access the resized tensor
resized_tensors
# %%
original_size = (5, 5)
sizes = [(3, 3), (5, 5), (7, 7), (3, 7), (7, 3)]
heatmaps = [torch.from_numpy(np.random.randint(0, 5, (x, y))).float() for x, y in sizes]
heatmaps
# %%
for heatmap in heatmaps:
    resized_heatmap = resize_tensor(heatmap, original_size)
    print(resized_heatmap)

# %%
heatmap = torch.from_numpy(np.random.randint(0, 5, (5, 6, 6))).float()
heatmap.shape
# %%
# add a channel dimension between batch and height
heatmap = heatmap.unsqueeze(1)
# %%
heatmap
# %%
resized_heatmap = F.interpolate(
    heatmap, size=original_size, mode="bilinear", align_corners=True
)
# %%
resized_heatmap.shape
# %%
resized_heatmap = resized_heatmap.squeeze()
# %%
resized_heatmap.shape
# %%
resized_heatmap
from typing import Tuple

# %%
from torch import Tensor


def interpolate_to_original_size(heatmap: Tensor, original_size: Tuple[int, int]):
    # Add a channel dimension between batch and height
    heatmap = heatmap.unsqueeze(1)
    # Resize using interpolate
    resized_heatmap = F.interpolate(
        heatmap, size=original_size, mode="bilinear", align_corners=True
    )
    # Remove added dimensions
    resized_heatmap = resized_heatmap.squeeze()
    return resized_heatmap


# %%
from torch import Tensor


def interpolate_to_size(
    heatmap: Tensor, size: Tuple[int, int], align_corners: bool = True
) -> Tensor:
    """
    Resizes a heatmap to its original size using bilinear interpolation.

    Parameters:
    - heatmap (Tensor): A Tensor representing the heatmap to be resized.
                         Expected shape: [batch_size, height, width].
    - original_size (Tuple[int, int]): A tuple representing the original size
                                       (height, width) to which the heatmap
                                       will be resized.

    Returns:
    - Tensor: A Tensor representing the resized heatmap. The returned heatmap
              maintains the batch size of the input but has its spatial dimensions
              resized to the original size.

    Note:
    - The function assumes the input heatmap has no channel dimension, and
      a channel dimension is temporarily added for the purpose of resizing.
    """

    add_batch_dim = heatmap.ndim == 2

    # Add a channel dimension between batch and height
    if add_batch_dim:
        heatmap = heatmap.unsqueeze(0)
    heatmap = heatmap.unsqueeze(1)
    # Resize using interpolate

    resized_heatmap = F.interpolate(
        heatmap, size=size, mode="bilinear", align_corners=align_corners
    )

    # Remove added dimensions
    resized_heatmap = resized_heatmap.squeeze(1)
    if add_batch_dim:
        resized_heatmap = resized_heatmap.squeeze(0)
    return resized_heatmap


# %%
heatmap = torch.from_numpy(np.random.randint(0, 5, (6, 6))).float()
heatmap
# %%
interpolate_to_size(heatmap, (5, 5))


# %%
def resize_heatmap_with_trim_and_pad(
    heatmap: Tensor, target_size: Tuple[int, int]
) -> Tensor:
    """
    Resizes a heatmap to a specified size by trimming or padding with zeros.

    Parameters:
    - heatmap (Tensor): A Tensor representing the heatmap to be resized.
                         Expected shape: [batch_size, height, width].
    - target_size (Tuple[int, int]): A tuple representing the target size
                                     (height, width) to which the heatmap
                                     will be resized.

    Returns:
    - Tensor: A Tensor representing the resized heatmap. The heatmap is trimmed
              if the target size is smaller than the original size, or padded
              with zeros if the target size is larger. The batch size of the
              input is maintained.

    Note:
    - The function does not change the channel dimension if present. It operates
      only on the spatial dimensions (height and width).
    """
    current_height, current_width = heatmap.shape[-2], heatmap.shape[-1]
    target_height, target_width = target_size

    # Determine padding or trimming for height
    if current_height < target_height:
        # Pad height
        pad_height_top = (target_height - current_height) // 2
        pad_height_bottom = target_height - current_height - pad_height_top
    else:
        # Trim height
        pad_height_top = 0
        pad_height_bottom = 0
        heatmap = heatmap[..., :target_height, :]

    # Determine padding or trimming for width
    if current_width < target_width:
        # Pad width
        pad_width_left = (target_width - current_width) // 2
        pad_width_right = target_width - current_width - pad_width_left
    else:
        # Trim width
        pad_width_left = 0
        pad_width_right = 0
        heatmap = heatmap[..., :target_width]

    # Apply padding
    heatmap = torch.nn.functional.pad(
        heatmap,
        (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom),
        "constant",
        0,
    )
    return heatmap


import numpy as np

# %%
import torch

# %%
heatmap = torch.from_numpy(np.random.randint(0, 5, (2, 6, 6))).float()
heatmap
# %%
new_heatmap = resize_heatmap_with_trim_and_pad(heatmap, (5, 5))
new_heatmap
# %%
new_heatmap.shape
# %%
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def upscale_heatmap(
    heatmap: Tensor, scale_factor: Union[int, Tuple[int, int]]
) -> Tensor:
    """
    Upscales a heatmap by a specified scale factor.

    Parameters:
    - heatmap (Tensor): A Tensor representing the heatmap to be upscaled.
                         Expected shape: [batch_size, height, width] or [height, width].
    - scale_factor (Union[int, Tuple[int, int]]): An integer or a tuple of two integers
                                                  representing the scale factor for
                                                  height and width. If an integer is
                                                  provided, the same scaling is applied
                                                  to both dimensions.

    Returns:
    - Tensor: A Tensor representing the upscaled heatmap. The spatial dimensions
              of the heatmap are scaled by the specified factor(s).

    Note:
    - This function uses bilinear interpolation for upscaling.
    - If the input heatmap lacks a batch dimension, it is temporarily added and
      removed in the output.
    """
    add_batch_dim = heatmap.ndim == 2

    # Add a batch dimension if necessary
    if add_batch_dim:
        heatmap = heatmap.unsqueeze(0)

    current_height, current_width = heatmap.shape[-2], heatmap.shape[-1]

    # Determine the scaling factors for height and width
    if isinstance(scale_factor, int):
        height_scale, width_scale = scale_factor, scale_factor
    else:
        height_scale, width_scale = scale_factor

    # Calculate new size
    new_height = current_height * height_scale
    new_width = current_width * width_scale

    # Upscale using interpolate
    upscaled_heatmap = F.interpolate(
        heatmap.unsqueeze(1),
        size=(new_height, new_width),
    ).squeeze(1)

    # Remove the batch dimension if it was added
    if add_batch_dim:
        upscaled_heatmap = upscaled_heatmap.squeeze(0)

    return upscaled_heatmap


# %%
heatmap = torch.from_numpy(np.random.randint(0, 5, (2, 3, 3))).float()
heatmap
# %%
new_heatmap = upscale_heatmap(heatmap, 2)
new_heatmap
# %%
