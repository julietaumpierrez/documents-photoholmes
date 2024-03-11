from photoholmes.preprocessing.image import GetImageSize, ToNumpy
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

dq_preprocessing = PreProcessingPipeline(
    transforms=[
        GetImageSize(),
        ToNumpy(),
        InputSelection(["dct_coefficients", "image_size"]),
    ]
)
