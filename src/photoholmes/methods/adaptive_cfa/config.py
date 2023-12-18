from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class AdaptiveCFAConfig:
    weights: Optional[Union[str, Path, dict]]
