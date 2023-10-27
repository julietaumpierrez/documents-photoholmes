from typing import Any


class PreprocessingTransform:
    def __call__(self, **kwargs) -> Any:
        raise NotImplementedError
