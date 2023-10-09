from dataclasses import dataclass


@dataclass
class WeightConfig:
    low_th: float = 6 / 255
    high_th: float = 252 / 255
    opening_kernel_radius: int = 3
    dilation_kernel_size: int = 9
