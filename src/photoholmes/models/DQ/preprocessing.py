from typing import Dict

from numpy.typing import NDArray
from torch import Tensor

from photoholmes.utils.image import tensor2numpy
from photoholmes.utils.preprocessing.base import PreprocessingTransform
from photoholmes.utils.preprocessing.pipeline import PreProcessingPipeline


class DQInput(PreprocessingTransform):
    def __call__(self, dct_coefficients: Tensor, **kwargs) -> Dict[str, NDArray]:
        np_dct = tensor2numpy(dct_coefficients)
        return {"dct_coefficients": np_dct.transpose(2, 0, 1)}


dq_preprocessing = PreProcessingPipeline(transforms=[DQInput()])
