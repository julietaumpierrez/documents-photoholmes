from photoholmes.preprocessing.base import PreprocessingTransform
from photoholmes.preprocessing.image import Normalize, RGBtoGray, ToNumpy
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

splicebuster_preprocess = PreProcessingPipeline(
    transforms=[ZeroOneRange(), RGBtoGray(), ToNumpy(), InputSelection(["image"])]
)
