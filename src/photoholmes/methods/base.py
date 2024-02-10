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
log = logging.getLogger("methods")
log.setLevel(logging.INFO)


T = TypeVar("T", NDArray, Tensor)


class BaseMethod(ABC):
    """Abstract class as a base for the methods"""

    @abstractmethod
    def __init__(self, threshold: float = 0.5, device: str = "cpu") -> None:
        """Initialization.
        he heatmap theshold value sets the default parameter for converting
        predicted heatmaps to masks in the "predict_mask" method.
        """
        self.threshold = threshold
        self.device = torch.device(device)

    @abstractmethod
    def predict(self, image: T) -> Dict[str, Any]:
        """Predicts heatmap from an image."""

    def predict_mask(self, heatmap):
        """Default strategy for mask prediction from the 'predict' (heatmap predicting)
        method.
        This method can be overriden for smarter post-processing algorythms,
            but should always use 'self.theshold' for metric evaluation purposes.
        """
        return heatmap > self.threshold

    @property
    def name(self):
        class_name = str(type(self)).split(".")[-1]
        return class_name[:-2]

    @classmethod
    def from_config(
        cls, config: Optional[str | Dict[str, Any]], device: Optional[str] = "cpu"
    ):
        """
        Instantiate the model from configuration dictionary or yaml.

        Params:
            config: path to the yaml configuration or a dictionary with
                    the parameters for the model.
            device: device to use for the model.
        """
        if isinstance(config, str):
            config = load_yaml(config)

        if config is None:
            config = {}

        config["device"] = device
        return cls(**config)

    def method_to_device(self, device: str):
        """Send the model to the device."""
        log.warning(
            f"Device wanted to be set to: `{device}`. "
            "Model does not implement 'model_to_device' method.\n"
            "Falling back to 'cpu' device."
        )
        self.device = torch.device("cpu")


class BaseTorchMethod(BaseMethod, Module):
    def __init__(self, threshold: float = 0.5, *args, **kwargs) -> None:
        Module.__init__(self, *args, **kwargs)
        BaseMethod.__init__(self, threshold=threshold)

    def load_weights(self, weights: Union[str, Path, dict]):
        if isinstance(weights, (str, Path)):
            weights_ = torch.load(weights, map_location=self.device)
        else:
            weights_ = weights

        if "state_dict" in weights_.keys():
            weights_ = weights_["state_dict"]

        self.load_state_dict(weights_, assign=True)
