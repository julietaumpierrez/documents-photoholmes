from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

# TODO: Cambiar Normalize por ZeroOneRange post mergear con develop
psccnet_preprocessing = PreProcessingPipeline(
    [
        ZeroOneRange(),
        InputSelection(["image", "original_image_size"]),
    ]
)
