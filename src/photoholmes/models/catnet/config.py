from typing import List, Literal, Optional, TypedDict


class StageConfig(TypedDict):
    NUM_CHANNELS: List[int]
    BLOCK: Literal["BOTTLENECK", "BASIC"]
    NUM_BLOCKS: List[int]
    NUM_BRANCHES: int
    NUM_MODULES: int
    FUSE_METHOD: Optional[Literal["SUM", "CAT"]]


class CatnetConfig(TypedDict):
    STAGE1: StageConfig
    STAGE2: StageConfig
    STAGE3: StageConfig
    STAGE4: StageConfig
    DC_STAGE3: StageConfig
    DC_STAGE4: StageConfig
    STAGE5: StageConfig
    FINAL_CONV_KERNEL: int
    BN_MOMENTUM: float


pretrain_config: CatnetConfig = {
    "FINAL_CONV_KERNEL": 1,
    "STAGE1": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 1,
        "BLOCK": "BOTTLENECK",
        "NUM_BLOCKS": [4],
        "NUM_CHANNELS": [64],
        "FUSE_METHOD": "SUM",
    },
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [48, 96],
        "FUSE_METHOD": "SUM",
    },
    "STAGE3": {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [48, 96, 192],
        "FUSE_METHOD": "SUM",
    },
    "STAGE4": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [48, 96, 192, 384],
        "FUSE_METHOD": "SUM",
    },
    "DC_STAGE3": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [96, 192],
        "FUSE_METHOD": "SUM",
    },
    "DC_STAGE4": {
        "NUM_MODULES": 2,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [96, 192, 384],
        "FUSE_METHOD": "SUM",
    },
    "STAGE5": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [24, 48, 96, 192],
        "FUSE_METHOD": "SUM",
    },
    "BN_MOMENTUM": 0.01,
}
