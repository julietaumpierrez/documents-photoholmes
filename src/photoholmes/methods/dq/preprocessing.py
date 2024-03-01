from photoholmes.preprocessing.image import ToNumpy
from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

dq_preprocessing = PreProcessingPipeline(
    transforms=[ToNumpy(), InputSelection(["dct_coefficients", "original_image_size"])]
)
