from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseMethod(ABC):
    """Abstract class as a base for the methods"""

    @abstractmethod
    def __init__(self, threshold: float = 0.5) -> None:
        """Initialization.
        he heatmap theshold value sets the default parameter for converting
        predicted heatmaps to masks in the "predict_mask" method.
        """
        self.threshold = threshold

    @classmethod
    @abstractmethod
    def from_config(cls, config: Optional[str | Dict[str, str]]):
        """Initializes model from a read config.
        By default, it takes 'config.yaml' in the model folder
        """

    @abstractmethod
    def predict(self, image):
        """Predicts heatmap from an image."""

    def predict_mask(self, heatmap):
        """Default strategy for mask prediction from the 'predict' (heatmap predicting) method.
        This method can be overriden for smarter post-processing algorythms,
            but should always use 'self.theshold' for metric evaluation purposes.
        """
        return heatmap > self.threshold

    @property
    def name(self):
        class_name = str(type(self)).split(".")[-1]
        return class_name[:-2]
