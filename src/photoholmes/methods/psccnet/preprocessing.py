from photoholmes.preprocessing.image import Normalize
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

# TODO: Cambiar Normalize por ZeroOneRange post mergear con develop
psccnet_preprocessing = PreProcessingPipeline(transforms=[Normalize()])
