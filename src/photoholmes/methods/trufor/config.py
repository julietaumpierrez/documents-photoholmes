from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union


@dataclass
class TruForConfig:
    backbone: Literal["mit_b2"] = "mit_b2"
    decoder: str = "MLPDecoder"
    num_classes: int = 2
    decoder_embed_dim: int = 512
    preprocess: str = "imagenet"
    bn_eps: float = 0.001
    bn_momentum: float = 0.01
    detection: Optional[str] = "confpool"
    confidence: bool = True
    mods: Sequence[Literal["NP++", "RGB"]] = ("NP++", "RGB")

    confidence_backbone: Optional[Literal["mit_b2"]] = None

    weights: Optional[Union[str, dict]] = None


PRETRAINED_CONFIG = TruForConfig()
