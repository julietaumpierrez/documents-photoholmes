from typing import Optional, Tuple, Union

from photoholmes.methods.base import BaseMethod
from photoholmes.methods.registry import MethodName
from photoholmes.preprocessing import PreProcessingPipeline


class MethodFactory:
    @staticmethod
    def load(
        method_name: Union[str, MethodName],
        config: Optional[Union[dict, str]] = None,
        device: Optional[str] = "cpu",
    ) -> Tuple[BaseMethod, PreProcessingPipeline]:
        """
        Instantiates and returns a method object along with its preprocessing pipeline,
        corresponding to the specified method name and configuration.

        Args:
            method_name (Union[str, MethodName]): The name of the method to load.
                Can be a string or a MethodName enum instance.
            config (Optional[Union[dict, str]]): The configuration for the method.
                Can be a dictionary of parameters or a string path to a config file.

        Returns:
            Tuple[BaseMethod, PreProcessingPipeline]: A tuple containing an instance of
                a subclass of photoholmes.methods.base.BaseMethod and its associated
                PreProcessingPipeline.

        Raises:
            NotImplementedError: If the method name provided is not recognized or not
                implemented.

        Examples:
            >>> method, preprocess = MethodFactory.load("naive")
            >>> method, preprocess = MethodFactory.load(MethodName.CATNET, config_dict)
        """
        if isinstance(method_name, str):
            method_name = MethodName(method_name.lower())
        match method_name:
            case MethodName.NAIVE:
                from photoholmes.methods.naive import Naive, naive_preprocessing

                return Naive.from_config(config, device), naive_preprocessing
            case MethodName.DQ:
                from photoholmes.methods.DQ import DQ, dq_preprocessing

                return DQ.from_config(config, device), dq_preprocessing
            case MethodName.SPLICEBUSTER:
                from photoholmes.methods.splicebuster import (
                    Splicebuster,
                    splicebuster_preprocess,
                )

                return Splicebuster.from_config(config, device), splicebuster_preprocess
            case MethodName.CATNET:
                from photoholmes.methods.catnet import CatNet, catnet_preprocessing

                return CatNet.from_config(config, device), catnet_preprocessing
            case MethodName.EXIF_AS_LANGUAGE:
                from photoholmes.methods.exif_as_language import (
                    EXIFAsLanguage,
                    exif_preprocessing,
                )

                return EXIFAsLanguage.from_config(config, device), exif_preprocessing
            case MethodName.ADAPTIVE_CFA_NET:
                from photoholmes.methods.adaptive_cfa_net import (
                    AdaptiveCFANet,
                    adaptive_cfa_net_preprocessing,
                )

                return (
                    AdaptiveCFANet.from_config(config),
                    adaptive_cfa_net_preprocessing,
                )
            case MethodName.NOISESNIFFER:
                from photoholmes.methods.noisesniffer import (
                    Noisesniffer,
                    noisesniffer_preprocess,
                )

                return Noisesniffer.from_config(config, device), noisesniffer_preprocess
            case MethodName.PSCCNET:
                from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing

                return PSCCNet.from_config(config), psccnet_preprocessing
            case MethodName.TRUFOR:
                from photoholmes.methods.trufor import TruFor, trufor_preprocessing

                return TruFor.from_config(config), trufor_preprocessing
            case _:
                raise NotImplementedError(f"Method '{method_name}' is not implemented.")
