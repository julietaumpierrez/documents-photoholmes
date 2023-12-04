from typing import Dict

from numpy.typing import NDArray

from photoholmes.preprocessing.base import PreprocessingTransform
from photoholmes.preprocessing.image import ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


class NoisesnifferInput(PreprocessingTransform):
    def __call__(self, image: NDArray, **kwargs) -> Dict[str, NDArray]:
        return {"image": image}


noisesniffer_preprocess = PreProcessingPipeline(
    transforms=[ToNumpy(), NoisesnifferInput()]
)
