from abc import ABC, abstractclassmethod

import numpy as np

from photoholmes.utils.generic import load_yaml

class BaseMethod(ABC):
    '''Abstract class as a base for the methods
    '''
    DEFAULT_CONFIG_PATH = 'photoholmes/models/'
    @abstractclassmethod
    def __init__(self, **kwargs) -> None:
        '''Initialization. 
        he heatmap theshold value sets the default parameter for converting predicted heatmaps to masks in the "predict_mask" method.
        '''
        if "threshold" in kwargs:
            self.theshold = kwargs["threshold"]

    @classmethod
    def from_config(cls, config:dict = None):
        '''Initializes model from a read config. 
        By default, it takes 'config.yaml' in the model folder
        '''
        if config is None:
            config = load_yaml(cls.DEFAULT_CONFIG_PATH)
        return cls(**config["default_kwargs"])

    @abstractclassmethod
    def predict(self, image):
        """Predicts heatmap from an image."""
        pass

    def predict_mask(self, heatmap):
        '''Default strategy for mask prediction from the 'predict' (heatmap predicting) method.
        This method can be overriden for smarter post-processing algorythms, 
            but should always use 'self.theshold' for metric evaluation purposes.
        '''
        return heatmap > self.theshold

    @property
    def name(self):
        class_name = str(type(self)).split(".")[-1]
        return class_name[:-2]
