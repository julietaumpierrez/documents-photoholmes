from typing import Any, Dict

import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor


def to_tensor_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts all the values in a dictionary to tensors.

    Args:
        input_dict: Dictionary to be converted to tensors.

    Returns:
        A dictionary with all the values converted to tensors.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, Tensor):
            output_dict[key] = value.float()
        elif isinstance(value, np.ndarray):
            output_dict[key] = torch.from_numpy(value).float()
        elif isinstance(value, Image):
            output_dict[key] = torch.from_numpy(np.array(value)).float()
        elif isinstance(value, (int, float)):
            output_dict[key] = torch.tensor(value).unsqueeze(0).float()
        else:
            output_dict[key] = value
    return output_dict


def to_numpy_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts all the values in a dictionary to numpy arrays.

    Args:
        input_dict: Dictionary to be converted to numpy arrays.

    Returns:
        A dictionary with all the values converted to numpy arrays.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            output_dict[key] = value
        elif isinstance(value, Tensor):
            output_dict[key] = value.cpu().numpy()
        elif isinstance(value, Image):
            output_dict[key] = np.array(value)
        else:
            output_dict[key] = value
    return output_dict


def zero_one_range(value: Any) -> Any:
    """
    Rescales the input value to the range [0, 1].

    Args:
        value: Value to be rescaled.

    Returns:
        The rescaled value.
    """
    if isinstance(value, Tensor):
        if value.dtype == torch.uint8:
            value = value.float() / 255
        elif value.max() > 1:
            value = value.float() / 255
        elif value.max() < 1:
            value = (value - value.min()) / (value.max() - value.min())
    elif isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            value = value.astype(np.float32) / 255
        elif value.max() > 1:
            value = value.astype(np.float32) / 255
        elif value.max() < 1:
            value = (value - value.min()) / (value.max() - value.min())
    return value
