import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module

from photoholmes.utils.generic import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


T = TypeVar("T", NDArray, Tensor)


class BaseMethod(ABC):
    """Abstract class as a base for the methods"""

    device: torch.device

    @abstractmethod
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    @abstractmethod
    def predict(self, image: T) -> Dict[str, Any]:
        """
        Runs method on an image.
        """

    @classmethod
    def from_config(cls, config: Optional[str | Dict[str, Any]]):
        """
        Instantiate the model from configuration dictionary or yaml.

        Params:
            config: path to the yaml configuration or a dictionary with
                    the parameters for the model.
        """
        if isinstance(config, str):
            config = load_yaml(config)

        if config is None:
            config = {}

        return cls(**config)

    def to_device(self, device: Union[str, torch.device]):
        """Send the model to the device."""
        log.warning(
            f"Device wanted to be set to: `{device}`. "
            "Model does not implement 'model_to_device' method.\n"
            "Falling back to 'cpu' device."
        )
        self.device = torch.device("cpu")


class BaseTorchMethod(BaseMethod, Module):
    def __init__(self, *args, **kwargs) -> None:
        Module.__init__(self, *args, **kwargs)
        BaseMethod.__init__(self)

    def load_weights(self, weights: Union[str, Path, dict]):
        if isinstance(weights, (str, Path)):
            weights_ = torch.load(weights, map_location=self.device)
        else:
            weights_ = weights

        if "state_dict" in weights_.keys():
            weights_ = weights_["state_dict"]

        self.load_state_dict(
            weights_, assign=True
        )  # FIXME: asign limits torch version to >=2.1

    def to_device(self, device: Union[str, torch.device]):
        self.to(self.device)
        self.device = torch.device(device)
