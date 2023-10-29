from abc import ABC, abstractmethod
from typing import Any


class PreprocessingTransform(ABC):
    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        pass
