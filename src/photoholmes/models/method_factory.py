from typing import Optional, Tuple, Union

from photoholmes.models.base import BaseMethod
from photoholmes.models.registry import MethodName
from photoholmes.utils.preprocessing import PreProcessingPipeline


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
                from photoholmes.models.naive.method import Naive

                return Naive.from_config(config), PreProcessingPipeline([])
            case MethodName.DQ:
                from photoholmes.models.DQ import DQ, dq_preprocessing

                return DQ.from_config(config), dq_preprocessing
            case MethodName.SPLICEBUSTER:
                from photoholmes.models.splicebuster import (
                    Splicebuster, splicebuster_preprocess)

                return Splicebuster.from_config(config), splicebuster_preprocess
            case MethodName.CATNET:
                from photoholmes.models.catnet import (CatNet,
                                                       catnet_preprocessing)

                return CatNet.from_config(config), catnet_preprocessing
            case _:
                raise NotImplementedError(f"Method '{method_name}' is not implemented.")
