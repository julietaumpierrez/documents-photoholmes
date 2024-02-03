from photoholmes.preprocessing.image import RGBtoYCrCb, ToNumpy
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

zero_preprocessing = PreProcessingPipeline(
    transforms=[RGBtoYCrCb(), ToNumpy(), InputSelection(["image"])]
)
