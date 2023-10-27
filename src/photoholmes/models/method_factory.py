from enum import Enum, unique
from typing import Optional, Tuple, Union

from photoholmes.models import Naive, Splicebuster
from photoholmes.models.base import BaseMethod
from photoholmes.models.DQ import DQ, dq_preprocessing
from photoholmes.models.splicebuster import (Splicebuster,
                                             splicebuster_preprocess)
from photoholmes.utils.preprocessing import PreProcessingPipeline


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
    ) -> Tuple[BaseMethod, PreProcessingPipeline]:
        """Instantiates method corresponding to the name passed, from config"""
        if isinstance(method_name, str):
            method_name = MethodName(method_name.lower())
        match method_name:
            case MethodName.NAIVE:
                return Naive.from_config(config), PreProcessingPipeline([])
            case MethodName.DQ:
                return (DQ.from_config(config), dq_preprocessing)
            case MethodName.SPLICEBUSTER:
                return Splicebuster.from_config(config), splicebuster_preprocess
            case _:
                raise NotImplementedError(f"Method '{method_name}' is not implemented.")
