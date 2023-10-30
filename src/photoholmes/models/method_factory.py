from enum import Enum, unique
from typing import Optional

from photoholmes.models.catnet import CatNet, catnet_preprocessing
from photoholmes.models.DQ import DQ, dq_preprocessing
from photoholmes.models.splicebuster import Splicebuster, splicebuster_preprocess


@unique
class MethodType(Enum):
    NAIVE = 0
    DQ = 1
    SPLICEBUSTER = 2
    CATNET = 3


def string_to_type(method_name: str) -> MethodType:
    if method_name == "naive":
        return MethodType.NAIVE
    elif method_name == "dq":
        return MethodType.DQ
    elif method_name == "splicebuster":
        return MethodType.SPLICEBUSTER
    elif method_name == "catnet":
        return MethodType.CATNET
    else:
        raise Exception("Method Type not implemented yet.")


class MethodFactory:
    @staticmethod
    def create(method_name: str, config: Optional[dict] = None):
        """Instantiates method corresponding to the name passed, from config"""
        method_type = string_to_type(method_name)
        if method_type == MethodType.NAIVE:
            return Naive.from_config(config)
        elif method_type == MethodType.DQ:
            return DQ.from_config(config), dq_preprocessing
        elif method_type == MethodType.SPLICEBUSTER:
            return Splicebuster.from_config(config), splicebuster_preprocess
        elif method_type == MethodType.CATNET:
            return CatNet.from_config(config), catnet_preprocessing
        else:
            raise Exception("Selected method_name is not defined")
