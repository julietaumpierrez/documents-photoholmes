from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module

from photoholmes.utils.generic import load_yaml

T = TypeVar("T", NDArray, Tensor)


class BaseMethod(ABC):
    """Abstract class as a base for the methods"""

    @abstractmethod
    def __init__(self, threshold: float = 0.5) -> None:
        """Initialization.
        he heatmap theshold value sets the default parameter for converting
        predicted heatmaps to masks in the "predict_mask" method.
        """
        self.threshold = threshold

    @abstractmethod
    def predict(self, image: T) -> T:
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


class BaseTorchMethod(BaseMethod, Module):
    def __init__(self, threshold: float = 0.5, *args, **kwargs) -> None:
        BaseMethod.__init__(self, threshold=threshold)
        Module.__init__(self, *args, **kwargs)

    def load_weigths(self, weights: Union[str, Path, dict]):
        if isinstance(weights, (str, Path)):
            # trick to get current device
            weights_ = torch.load(
                weights, map_location=next(self.parameters())[0].device
            )
        else:
            weights_ = weights

        if "state_dict" in weights_.keys():
            weights_ = weights_["state_dict"]

        self.load_state_dict(weights_)
