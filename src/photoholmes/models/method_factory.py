from enum import Enum, unique
from typing import Optional

from .naive.method import Naive


@unique
class MethodType(Enum):
    NAIVE = 0


def string_to_type(method_name: str) -> MethodType:
    if method_name == "naive":
        return MethodType.NAIVE
    else:
        raise Exception("Method Type not implemented yet.")


class MethodFactory:
    @staticmethod
    def create(method_name: str, config: Optional[dict] = None):
        """Instantiates method corresponding to the name passed, from config"""
        method_type = string_to_type(method_name)
        if method_type == MethodType.NAIVE:
            return Naive.from_config(config)
        else:
            raise Exception("Selected method_name is not defined")
