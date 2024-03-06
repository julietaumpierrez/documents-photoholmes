from dataclasses import dataclass


@dataclass
class WeightConfig:
    low_th: int = 6
    high_th: int = 252
    erotion_kernel_size: int = 9
