from typing import Dict

from numpy.typing import NDArray

from photoholmes.utils.preprocessing.base import (
    PreProcessingPipeline,
    PreprocessingTransform,
)
from photoholmes.utils.preprocessing.image import Normalize, RGBtoGray


class SplicebusterInput(PreprocessingTransform):
    def __call__(self, image: NDArray, **kwargs) -> Dict[str, NDArray]:
        return {"image": image}


splicebuster_preprocess = PreProcessingPipeline(
    transforms=[Normalize(), RGBtoGray(), SplicebusterInput()]
)
