from pathlib import Path
from typing import Optional, Tuple, Union

from photoholmes.methods.base import BaseMethod
from photoholmes.methods.registry import MethodName
from photoholmes.preprocessing import PreProcessingPipeline


class MethodFactory:
    """
    MethodFactory class responsible for loading methods and their associated
    preprocessing pipelines.

    Supported methods:
        - AdaptiveCFANet
        - CatNet
        - DQ
        - EXIFAsLanguage
        - Focal
        - Naive
        - Noisesniffer
        - PSCCNet
        - Splicebuster
        - TruFor
        - Zero

    Methods:
        load(method_name: Union[str, MethodName], config: Optional[Union[dict, str]] =
            None) -> Tuple[BaseMethod, PreProcessingPipeline]:
            Instantiates and returns a method object along with its preprocessing
            pipeline, corresponding to the specified method name and configuration.
    """

    @staticmethod
    def load(
        method_name: Union[str, MethodName],
        config: Optional[Union[dict, str, Path]] = None,
    ) -> Tuple[BaseMethod, PreProcessingPipeline]:
        """
        Instantiates and returns a method object along with its preprocessing pipeline,
        corresponding to the specified method name and configuration.

        Args:
            method_name (Union[str, MethodName]): the name of the method to load.
                Can be a string or a MethodName enum.

            config (Optional[Union[dict, str, Path]]): the configuration to use when
                instantiating the method. Can be a dictionary, a string representing a
                path to a configuration file, or a Path object. Defaults to None.

        Returns:
            Tuple[BaseMethod, PreProcessingPipeline]: a tuple containing the method
            object and its preprocessing pipeline.

        Raises:
            NotImplementedError: If the method name provided is not recognized or not
                implemented.

        Examples:
            # Load the Naive method with default configuration.
            >>> method, preprocess = MethodFactory.load("naive")

            # Load the CatNet method the MethodName enum and a configuration file.
            >>> method, preprocess = MethodFactory.load(MethodName.CATNET,
                                        "path/to/config.yaml")
        """
        if isinstance(method_name, str):
            method_name = MethodName(method_name.lower())
        match method_name:
            case MethodName.ADAPTIVE_CFA_NET:
                from photoholmes.methods.adaptive_cfa_net import (
                    AdaptiveCFANet,
                    adaptive_cfa_net_preprocessing,
                )

                return (
                    AdaptiveCFANet.from_config(config),
                    adaptive_cfa_net_preprocessing,
                )
            case MethodName.CATNET:
                from photoholmes.methods.catnet import CatNet, catnet_preprocessing

                return CatNet.from_config(config), catnet_preprocessing
            case MethodName.DQ:
                from photoholmes.methods.dq import DQ, dq_preprocessing

                return DQ.from_config(config), dq_preprocessing
            case MethodName.EXIF_AS_LANGUAGE:
                from photoholmes.methods.exif_as_language import (
                    EXIFAsLanguage,
                    exif_preprocessing,
                )

                return EXIFAsLanguage.from_config(config), exif_preprocessing
            case MethodName.FOCAL:
                from photoholmes.methods.focal import Focal, focal_preprocessing

                return Focal.from_config(config), focal_preprocessing
            case MethodName.NAIVE:
                from photoholmes.methods.naive import Naive, naive_preprocessing

                return Naive.from_config(config), naive_preprocessing
            case MethodName.NOISESNIFFER:
                from photoholmes.methods.noisesniffer import (
                    Noisesniffer,
                    noisesniffer_preprocess,
                )

                return Noisesniffer.from_config(config), noisesniffer_preprocess
            case MethodName.PSCCNET:
                from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing

                return PSCCNet.from_config(config), psccnet_preprocessing
            case MethodName.SPLICEBUSTER:
                from photoholmes.methods.splicebuster import (
                    Splicebuster,
                    splicebuster_preprocess,
                )

                return Splicebuster.from_config(config), splicebuster_preprocess
            case MethodName.TRUFOR:
                from photoholmes.methods.trufor import TruFor, trufor_preprocessing

                return TruFor.from_config(config), trufor_preprocessing
            case MethodName.ZERO:
                from photoholmes.methods.zero import Zero, zero_preprocessing

                return Zero.from_config(config), zero_preprocessing
            case _:
                raise NotImplementedError(f"Method '{method_name}' is not implemented.")
