from abc import ABC, abstractclassmethod
import numpy as np

class Method(ABC):
    '''Abstract class as a base for the methods
    '''
    @abstractclassmethod
    def __init__(self, heatmap_threshold:float = 0.5) -> None:
        '''Initialization. 
        he heatmap theshold value sets the default parameter for converting predicted heatmaps to masks in the "predict_mask" method.
        '''
        self.theshold = heatmap_threshold

    @classmethod
    def from_config(cls, config):
        return cls()

    @abstractclassmethod
    def predict(self, image):
        '''Predicts heatmap from an image.
        '''
        pass

    def predict_mask(self, image):
        '''Default strategy for mask prediction from the 'predict' (heatmap predicting) method.
        This method can be overriden for smarter post-processing algorythms, 
            but should always use 'self.theshold' for metric evaluation purposes.
        '''
        return self.predict(image) > self.theshold

    @property
    def name(self):
        class_name = str(type(self)).split('.')[-1]
        return class_name[:-2]