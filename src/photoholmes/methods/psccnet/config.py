from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union


@dataclass
class StageConfig:
    num_modules: int
    num_branches: int
    num_blocks: List[int]
    num_channels: List[int]
    block: Literal["BOTTLENECK", "BASIC"]
    fuse_method: Literal["SUM"]


@dataclass
class PSCCArchConfig:
    stage1: StageConfig
    stage2: StageConfig
    stage3: StageConfig
    stage4: StageConfig
    stem_inplanes: int = 64
    final_conv_kernel: int = 1

    @classmethod
    def load_from_dict(cls, config_dict: dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                config_dict[k] = StageConfig(**v)
            else:
                config_dict[k] = v
        return cls(**config_dict)


pretrained_arch = PSCCArchConfig(
    final_conv_kernel=1,
    stem_inplanes=64,
    stage1=StageConfig(
        num_modules=1,
        num_branches=1,
        num_blocks=[2],
        num_channels=[64],
        block="BOTTLENECK",
        fuse_method="SUM",
    ),
    stage2=StageConfig(
        num_modules=1,
        num_branches=2,
        num_blocks=[2, 2],
        num_channels=[18, 36],
        block="BASIC",
        fuse_method="SUM",
    ),
    stage3=StageConfig(
        num_modules=1,
        num_branches=3,
        num_blocks=[2, 2, 2],
        num_channels=[18, 36, 72],
        block="BASIC",
        fuse_method="SUM",
    ),
    stage4=StageConfig(
        num_modules=1,
        num_branches=4,
        num_blocks=[2, 2, 2, 2],
        num_channels=[18, 36, 72, 144],
        block="BASIC",
        fuse_method="SUM",
    ),
)


@dataclass
class PSCCConfig:
    weights: Optional[Union[str, Path, dict]]
    arch_config: Union[PSCCArchConfig, Literal["pretrained"]] = "pretrained"
