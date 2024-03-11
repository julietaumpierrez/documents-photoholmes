from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

focal_preproceesing = PreProcessingPipeline(
    transforms=[ZeroOneRange()],
    inputs=["image"],
    outputs_keys=["image"],
)
