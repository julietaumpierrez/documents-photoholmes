from typing import Any, Dict, List

from photoholmes.preprocessing.base import PreprocessingTransform


class PreProcessingPipeline:
    """
    A pipeline of preprocessing transforms.

    Args:
        transforms: A list of preprocessing transforms to apply sequentially to the
          input.

    Returns:
        A dictionary with the output of the last transform in the pipeline.
    """

    def __init__(self, transforms: List[PreprocessingTransform]) -> None:
        """
        Initializes a new preprocessing pipeline.

        Args:
            transforms: A list of preprocessing transforms to apply to the input.
        """
        self.transforms = transforms

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Applies the preprocessing pipeline to the input.

        Args:
            **kwargs: Keyword arguments representing the input to the pipeline.

        Returns:
            A dictionary with the output of the last transform in the pipeline.
        """
        for t in self.transforms:
            kwargs = t(**kwargs)

        return kwargs

    def add(self, transform: PreprocessingTransform):
        self.transforms.append(transform)

    def insert(self, transform: PreprocessingTransform, index: int):
        self.transforms.insert(index, transform)
