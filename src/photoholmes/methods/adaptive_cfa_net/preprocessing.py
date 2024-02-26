from typing import Any, Dict

from torch import Tensor

from photoholmes.preprocessing.base import PreprocessingTransform
from photoholmes.preprocessing.image import GetImageSize, ZeroOneRange
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


class AdaptiveCFANetPreprocessing(PreprocessingTransform):
    def __init__(self):
        pass

    def __call__(
        self,
        image: Tensor,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        C, Y_o, X_o = image.shape
        image = image[:C, : Y_o - Y_o % 2, : X_o - X_o % 2]

        return {"image": image, **kwargs}


adaptive_cfa_net_preprocessing = PreProcessingPipeline(
    [
        GetImageSize(),
        ZeroOneRange(),
        AdaptiveCFANetPreprocessing(),
        InputSelection(["image", "image_size"]),
    ]
)
