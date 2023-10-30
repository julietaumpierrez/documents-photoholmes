from typing import Dict

from numpy.typing import NDArray

from photoholmes.utils.preprocessing.base import PreprocessingTransform
from photoholmes.utils.preprocessing.pipeline import PreProcessingPipeline


class DQInput(PreprocessingTransform):
    def __call__(self, dct_coefficients: NDArray, **kwargs) -> Dict[str, NDArray]:
        return {"dct_coefficients": dct_coefficients}


dq_preprocessing = PreProcessingPipeline(transforms=[DQInput()])
