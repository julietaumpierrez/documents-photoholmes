from photoholmes.preprocessing.image import Normalize
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

psccnet_preprocessing = PreProcessingPipeline(transforms=[Normalize()])
