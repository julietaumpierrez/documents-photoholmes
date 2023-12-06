from photoholmes.preprocessing.image import ToNumpy
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

naive_preprocessing = PreProcessingPipeline(
    transforms=[ToNumpy(), InputSelection(["image"])]
)
