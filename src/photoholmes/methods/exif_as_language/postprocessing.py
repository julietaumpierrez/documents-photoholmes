from typing import Any, Dict, Union

import torch

from photoholmes.postprocessing.device import to_device_dict
from photoholmes.postprocessing.image import to_tensor_dict, zero_one_range


def exif_as_language_postprocessing(
    input_dict: Dict[str, Any], device: Union[str, torch.device]
) -> Dict[str, Any]:
    """
    Postprocessing function for the ExifAsLanguage method.

    Args:
        input_dict: Input dictionary.
        device: Device to use.

    Returns:
        A dictionary with the postprocessed output.
    """

    output_dict = to_device_dict(input_dict, device)
    output_dict["heatmap"] = (
        zero_one_range(output_dict["heatmap"]) * output_dict["detection"]
    )
    output_dict = to_tensor_dict(output_dict)

    return output_dict
