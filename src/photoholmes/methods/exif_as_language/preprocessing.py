from photoholmes.preprocessing import PreProcessingPipeline
from photoholmes.preprocessing.image import GrayToRGB, Normalize, ZeroOneRange
from photoholmes.preprocessing.input import InputSelection

exif_preprocessing = PreProcessingPipeline(
    [
        ZeroOneRange(),
        GrayToRGB(),
        Normalize(
            # We don't know were they got this value, an issue was
            # opened in their repo: https://github.com/hellomuffin/exif-as-language/issues/9
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
        InputSelection(["image"]),
    ]
)
