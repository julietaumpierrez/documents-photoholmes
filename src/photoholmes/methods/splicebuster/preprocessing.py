from photoholmes.preprocessing.image import RGBtoGray, ToNumpy, ZeroOneRange
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

splicebuster_preprocess = PreProcessingPipeline(
    transforms=[ZeroOneRange(), RGBtoGray(), ToNumpy(), InputSelection(["image"])]
)
