from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

psccnet_preprocessing = PreProcessingPipeline(
    [
        ZeroOneRange(),
        InputSelection(["image"]),
    ]
)
