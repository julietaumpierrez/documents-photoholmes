from typing import Any, Dict, List


class BaseTransform:
    def __call__(self, **kwargs) -> Any:
        raise NotImplementedError
