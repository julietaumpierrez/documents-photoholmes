from typing import Dict

from numpy.typing import NDArray

from photoholmes.preprocessing.base import PreprocessingTransform
from photoholmes.preprocessing.image import Normalize, RGBtoGray, ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


class SplicebusterInput(PreprocessingTransform):
    def __call__(self, image: NDArray, **kwargs) -> Dict[str, NDArray]:
        return {"image": image}


splicebuster_preprocess = PreProcessingPipeline(
    transforms=[ZeroOneRange(), RGBtoGray(), ToNumpy(), SplicebusterInput()]
)
