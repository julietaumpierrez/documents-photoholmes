from typing import List

from photoholmes.preprocessing.base import PreprocessingTransform


class InputSelection(PreprocessingTransform):
    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, **kwargs):
        return {k: v for k, v in kwargs.items() if k in self.keys}
