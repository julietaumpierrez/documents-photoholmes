from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union


@dataclass
class DirFullDirConfig:
    n_dir: int
    n_full: int
    n_dir_dil: int
    n_full_dil: int


@dataclass
class SkipDoubleDirFullDirConfig:
    channels_in: int
    convolutions_1: DirFullDirConfig
    convolutions_2: DirFullDirConfig


@dataclass
class PixelwiseConfig:
    conv1_in_channels: int
    conv1_out_channels: int
    conv2_out_channels: int
    conv3_out_channels: int
    conv4_out_channels: int
    kernel_size: int


@dataclass
class BlockwiseLayerConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    groups: int
    activation: Literal["Softplus", "LogSoftmax"]


@dataclass
class BlockwiseConfig:
    layers: List[BlockwiseLayerConfig]


@dataclass
class AdaptiveCFANetArchConfig:
    skip_double_dir_full_dir_config: SkipDoubleDirFullDirConfig
    pixelwise_config: PixelwiseConfig
    blockwise_config: BlockwiseConfig

    @classmethod
    def load_from_dict(cls, config_dict: dict):
        parsed_config = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                config_class_name = key.capitalize() + "Config"
                if config_class_name in globals():
                    config_class = globals()[config_class_name]
                    parsed_config[key] = (
                        config_class.load_from_dict(value)
                        if hasattr(config_class, "load_from_dict")
                        else config_class(**value)
                    )
                else:
                    parsed_config[key] = value  # Fallback if no matching class found
            else:
                parsed_config[key] = value
        return cls(**parsed_config)


pretrained_arch = AdaptiveCFANetArchConfig(
    skip_double_dir_full_dir_config=SkipDoubleDirFullDirConfig(
        channels_in=3,
        convolutions_1=DirFullDirConfig(n_dir=10, n_full=5, n_dir_dil=10, n_full_dil=5),
        convolutions_2=DirFullDirConfig(n_dir=10, n_full=5, n_dir_dil=10, n_full_dil=5),
    ),
    pixelwise_config=PixelwiseConfig(
        conv1_in_channels=103,
        conv1_out_channels=30,
        conv2_out_channels=15,
        conv3_out_channels=15,
        conv4_out_channels=30,
        kernel_size=1,
    ),
    blockwise_config=BlockwiseConfig(
        layers=[
            BlockwiseLayerConfig(
                in_channels=120,
                out_channels=180,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=30,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=180,
                out_channels=90,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=30,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=90,
                out_channels=90,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=30,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=90,
                out_channels=45,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=45,
                out_channels=45,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=45,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                activation="LogSoftmax",
            ),
        ]
    ),
)


@dataclass
class AdaptiveCFANetConfig:
    weights: Optional[Union[str, Path, dict]]
    arch: Union[AdaptiveCFANetArchConfig, Literal["pretrained"]] = "pretrained"
    device: str = "cpu"
