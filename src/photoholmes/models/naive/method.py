import numpy as np
import skimage as ski

from photoholmes.models.base import Method

class Naive(Method):
    '''A random method to test the program structure
    '''
    def __init__(self,):
        super().__init__()
    
    def predict(self, image:np.ndarray) -> np.ndarray:
        '''Predicts masks from a list of images.
        '''
        shape = image.shape[:2] if image.ndim>2 else image.shape
        return np.random.normal(0.5, 2, size=shape)
    