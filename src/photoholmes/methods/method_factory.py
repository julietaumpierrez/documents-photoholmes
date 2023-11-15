from typing import Optional, Tuple, Union

from photoholmes.methods.base import BaseMethod
from photoholmes.methods.registry import MethodName
from photoholmes.preprocessing import PreProcessingPipeline


class MethodFactory:
    @staticmethod
    def load(
        method_name: Union[str, MethodName],
        config: Optional[Union[dict, str]] = None,
    ) -> Tuple[BaseMethod, PreProcessingPipeline]:
        """Instantiates methods corresponding to the name passed, from config"""
        if isinstance(method_name, str):
            method_name = MethodName(method_name.lower())
        match method_name:
            case MethodName.NAIVE:
                from photoholmes.methods.naive.method import Naive

                return Naive.from_config(config), PreProcessingPipeline([])
            case MethodName.DQ:
                from photoholmes.methods.DQ import DQ, dq_preprocessing

                return DQ.from_config(config), dq_preprocessing
            case MethodName.SPLICEBUSTER:
                from photoholmes.methods.splicebuster import (
                    Splicebuster,
                    splicebuster_preprocess,
                )

                return Splicebuster.from_config(config), splicebuster_preprocess
            case MethodName.CATNET:
                from photoholmes.methods.catnet import CatNet, catnet_preprocessing

                return CatNet.from_config(config), catnet_preprocessing
            case _:
                raise NotImplementedError(f"Method '{method_name}' is not implemented.")
