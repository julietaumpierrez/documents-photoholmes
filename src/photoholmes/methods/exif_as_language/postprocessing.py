from typing import Any, Dict, Union

import torch

from photoholmes.postprocessing.device import to_device_dict
from photoholmes.postprocessing.image import to_tensor_dict


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
    output_dict = to_tensor_dict(input_dict)
    output_dict = to_device_dict(output_dict, device)

    return output_dict
