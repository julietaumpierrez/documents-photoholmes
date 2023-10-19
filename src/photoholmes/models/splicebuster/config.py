from dataclasses import dataclass


@dataclass
class WeightConfig:
    low_th: int = 6
    high_th: int = 252
    opening_kernel_radius: int = 3
    dilation_kernel_size: int = 9
