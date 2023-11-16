from typing import Any, Dict

from torch import Tensor

from photoholmes.utils.preprocessing.base import PreprocessingTransform
from photoholmes.utils.preprocessing.image import ToTensor
from photoholmes.utils.preprocessing.pipeline import PreProcessingPipeline


class CFANetPreprocessing(PreprocessingTransform):
    def __init__(self):
        pass

    def __call__(self, image: Tensor, **kwargs: Dict[str, Any]) -> Dict[str, Tensor]:
        C, Y_o, X_o = image.shape
        image = image[:C, : Y_o - Y_o % 2, : X_o - X_o % 2]

        image = image.float()
        if image.ndim == 3:
            image = image[None, :]

        return {"x": image}


cfanet_preprocessing = PreProcessingPipeline([ToTensor(), CFANetPreprocessing()])
