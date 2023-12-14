from typing import Any, Dict, Tuple

from torch import Tensor

from photoholmes.preprocessing.base import PreprocessingTransform
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


# TODO: check if normalization is needed
class AdaptiveCFANetPreprocessing(PreprocessingTransform):
    def __init__(self):
        pass

    def __call__(
        self,
        image: Tensor,
        original_image_size=Tuple[int, int],
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        C, Y_o, X_o = image.shape
        image = image[:C, : Y_o - Y_o % 2, : X_o - X_o % 2]

        image = image.float()

        return {"image": image, "original_image_size": original_image_size}


adaptive_cfa_net_preprocessing = PreProcessingPipeline([AdaptiveCFANetPreprocessing()])
