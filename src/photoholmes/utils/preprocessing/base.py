from typing import Any, Dict, List


class BaseTransform:
    def __call__(self, **kwargs) -> Any:
        raise NotImplementedError


class PreProcessingPipeline:
    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, **kwargs) -> Dict[str, Any]:
        for t in self.transforms:
            kwargs = t(**kwargs)

        return kwargs
