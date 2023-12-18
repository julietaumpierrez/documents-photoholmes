from photoholmes.preprocessing.image import ToNumpy
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

noisesniffer_preprocess = PreProcessingPipeline(
    transforms=[ToNumpy(), InputSelection(["image"])]
)
