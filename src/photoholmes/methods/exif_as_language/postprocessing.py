from typing import Union

import torch

from photoholmes.methods.base import BenchmarkOutput
from photoholmes.postprocessing.device import to_device_dict
from photoholmes.postprocessing.image import to_tensor_dict


def exif_as_language_postprocessing(
    input_dict: BenchmarkOutput, device: Union[str, torch.device]
) -> BenchmarkOutput:
    """
    Postprocessing function for the ExifAsLanguage method.

    Args:
        input_dict (BenchmarkOutput): The input dictionary to postprocess.
        device (Union[str, torch.device]): The device to use for postprocessing.

    Returns:
        BenchmarkOutput: The postprocessed dictionary.
    """
    output_dict = to_tensor_dict(input_dict)
    output_dict = to_device_dict(output_dict, device)

    return output_dict
