from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from photoholmes.models import DQ, Naive, Splicebuster


@unique
class MethodName(Enum):
    NAIVE = "naive"
    DQ = "dq"
    SPLICEBUSTER = "splicebuster"


class MethodFactory:
    @staticmethod
    def load(
        method_name: Union[str, MethodName],
        config: Optional[Union[dict, str]] = None,
    ):
        """Instantiates method corresponding to the name passed, from config"""
        if isinstance(method_name, str):
            method_name = MethodName(method_name.lower())
        match method_name:
            case MethodName.NAIVE:
                return Naive.from_config(config)
            case MethodName.DQ:
                return DQ.from_config(config)
            case MethodName.SPLICEBUSTER:
                return Splicebuster.from_config(config)
            case _:
                raise Exception(f"Method '{method_name}' is not implemented.")
