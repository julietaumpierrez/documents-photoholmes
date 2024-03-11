from photoholmes.preprocessing.image import ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

noisesniffer_preprocess = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image"],
    transforms=[ToNumpy()],
)
