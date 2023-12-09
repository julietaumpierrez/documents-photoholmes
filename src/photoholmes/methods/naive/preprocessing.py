from photoholmes.preprocessing.input import InputSelection
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

naive_preprocessing = PreProcessingPipeline(
    transforms=[InputSelection(["original_image_size"])]
)
