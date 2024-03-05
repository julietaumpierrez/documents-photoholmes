from photoholmes.preprocessing.image import RGBtoGray, RoundToUInt, ToNumpy
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

zero_preprocessing = PreProcessingPipeline(
    transforms=[RGBtoGray(), RoundToUInt(), ToNumpy(), InputSelection(["image"])]
)
