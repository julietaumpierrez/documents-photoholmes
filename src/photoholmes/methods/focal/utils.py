from typing import Any, Dict, Union

import torch
import torch.nn as nn


def load_weights(
    model: nn.Module, weights: Union[str, Dict[str, Any]], device: str = "cpu"
):
    if isinstance(weights, str):
        weights_ = torch.load(weights, map_location=torch.device(device))
    else:
        weights_ = weights

    model.load_state_dict(weights_)
