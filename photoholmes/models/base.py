from abc import ABC, abstractclassmethod
import numpy as np

class Method(ABC):
    '''Abstract class as a base for the methods
    '''
    @abstractclassmethod
    def __init__(self) -> None:
        '''Initialization
        '''
        super().__init__()
        pass

    @abstractclassmethod
    def predict_img(self, image:np.ndarray) -> np.ndarray:
        '''Predicts mask from an image.
        '''
        pass

    def predict(self, images:list[np.ndarray]) -> list[np.ndarray]:
        '''Predicts masks from a list of images, and evaluates them to a ground truth according to a metric
        '''
        return [self.predict_img(image) for image in images]
    
    @property
    def name(self):
        class_name = str(type(self)).split('.')[-1]
        return class_name[:-2]