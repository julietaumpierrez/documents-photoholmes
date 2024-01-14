from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class TruForConfig:
    backbone: Literal["mit_b2"]
    decoder: str
    num_classes: int
    decoder_embed_dim: int
    preprocess: str
    bn_eps: float
    bn_momentum: float
    detection: Optional[str]
    confidence: bool
    mods: List[Literal["NP++", "RGB"]]

    confidence_backbone: Optional[Literal["mit_b2"]] = None

    pretrained: Optional[str] = None


PRETRAINED_CONFIG = TruForConfig(
    backbone="mit_b2",
    decoder="MLPDecoder",
    num_classes=2,
    decoder_embed_dim=512,
    preprocess="imagenet",
    bn_eps=0.001,
    bn_momentum=0.01,
    detection="confpool",
    confidence=True,
    mods=["NP++", "RGB"],
)
