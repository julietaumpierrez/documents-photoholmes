# adapted from https://github.com/grip-unina/TruFor/blob/main/test_docker/src/models/cmx/builder_np_conf.py # noqa: E501
"""
Edited in September 2022
@author: fabrizio.guillaro, davide.cozzolino
"""

import logging
from typing import Any, Dict, Literal, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from photoholmes.methods.base import BaseTorchMethod
from photoholmes.utils.generic import load_yaml

from .config import PRETRAINED_CONFIG, TruForConfig
from .models.DnCNN import ActivationOptions, make_net
from .models.utils.init_func import init_weight
from .models.utils.layer import weighted_statistics_pooling

logger = logging.getLogger(__name__)


def preprc_imagenet_torch(x):
    """
    Imagenet preprocessing.
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    return x


def create_backbone(typ: Literal["mit_b2"], norm_layer: Type[nn.Module]):
    """
    Create backbone for trufor method.
    """
    channels = [64, 128, 320, 512]
    if typ == "mit_b2":
        logger.info("Using backbone: Segformer-B2")
        from .models.cmx.encoders.dual_segformer import mit_b2 as backbone_

        backbone = backbone_(norm_fuse=norm_layer)
    else:
        raise NotImplementedError(f"backbone `{typ}` not implemented")
    return backbone, channels


class TruFor(BaseTorchMethod):
    def __init__(
        self,
        cfg: Union[TruForConfig, Literal["pretrained"]] = "pretrained",
        norm_layer=nn.BatchNorm2d,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if cfg == "pretrained":
            cfg = PRETRAINED_CONFIG

        self.cfg = cfg

        self.norm_layer = norm_layer
        self.mods = cfg.mods

        # import backbone and decoder
        self.backbone, self.channels = create_backbone(self.cfg.backbone, norm_layer)

        if self.cfg.confidence_backbone is not None:
            self.confidence_backbone, self.channels_conf = create_backbone(
                self.cfg.confidence_backbone, norm_layer
            )
        else:
            self.confidence_backbone = None

        if self.cfg.decoder == "MLPDecoder":
            logger.info("Using MLP Decoder")
            from .models.cmx.decoders.MLPDecoder import DecoderHead

            self.decode_head = DecoderHead(
                in_channels=self.channels,
                num_classes=self.cfg.num_classes,
                norm_layer=norm_layer,
                embed_dim=self.cfg.decoder_embed_dim,
            )

            self.decode_head_conf: Optional[nn.Module]
            if self.cfg.confidence:
                self.decode_head_conf = DecoderHead(
                    in_channels=self.channels,
                    num_classes=1,
                    norm_layer=norm_layer,
                    embed_dim=self.cfg.decoder_embed_dim,
                )
            else:
                self.decode_head_conf = None

            self.conf_detection = None
            if self.cfg.detection is not None:
                if self.cfg.detection is None:
                    pass

                elif self.cfg.detection == "confpool":
                    self.conf_detection = "confpool"
                    assert self.cfg.confidence
                    self.detection = nn.Sequential(
                        nn.Linear(in_features=8, out_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=128, out_features=1),
                    )
                else:
                    raise NotImplementedError("Detection mechanism not implemented")

        else:
            raise NotImplementedError("decoder not implemented")

        num_levels = 17
        out_channel = 1
        npp_activations = [ActivationOptions.RELU] * (num_levels - 1) + [
            ActivationOptions.LINEAR
        ]
        self.dncnn = make_net(
            3,
            kernels=[
                3,
            ]
            * num_levels,
            features=[
                64,
            ]
            * (num_levels - 1)
            + [out_channel],
            bns=[
                False,
            ]
            + [
                True,
            ]
            * (num_levels - 2)
            + [
                False,
            ],
            acts=npp_activations,
            dilats=[
                1,
            ]
            * num_levels,
            bn_momentum=0.1,
            padding=1,
        )

        if self.cfg.preprocess == "imagenet":  # RGB (mean and variance)
            self.prepro = preprc_imagenet_torch
        else:
            assert False

        if cfg.weights is not None:
            self.load_weights(cfg.weights)
        else:
            logger.warn("No weight file provided. Initiralizing random weights.")
            self.init_weights()

        self.method_to_device(device)

    def init_weights(self):
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            self.cfg.bn_eps,
            self.cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )

    def encode_decode(self, rgb, modal_x):
        if rgb is not None:
            orisize = rgb.shape
        else:
            orisize = modal_x.shape

        # cmx
        x = self.backbone(rgb, modal_x)
        out, _ = self.decode_head(x, return_feats=True)
        out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)

        # confidence
        if self.decode_head_conf is not None:
            if self.confidence_backbone is not None:
                x_conf = self.confidence_backbone(rgb, modal_x)
            else:
                x_conf = x  # same encoder of Localization Network

            conf = self.decode_head_conf(x_conf)
            conf = F.interpolate(
                conf, size=orisize[2:], mode="bilinear", align_corners=False
            )
        else:
            conf = None

        # detection
        if self.conf_detection is not None and conf is not None:
            if self.conf_detection == "confpool":
                f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
                f2 = weighted_statistics_pooling(
                    out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)
                ).view(out.shape[0], -1)
                det = self.detection(torch.cat((f1, f2), -1))
            else:
                assert False
        else:
            det = None

        return out, conf, det

    def forward(self, rgb: torch.Tensor):
        # Noiseprint++ extraction
        if "NP++" in self.mods:
            modal_x = self.dncnn(rgb)
            modal_x = torch.tile(modal_x, (3, 1, 1))
        else:
            modal_x = None

        if self.prepro is not None:
            rgb = self.prepro(rgb)

        out, conf, det = self.encode_decode(rgb, modal_x)
        return out, conf, det, modal_x

    def predict(self, image: torch.Tensor):
        self.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            out, conf, det, npp = self.forward(image)

        # select the map with the smallest sum (smallest anomaly area)
        sum_maps = torch.sum(out, dim=[-1, -2])
        heatmap = out[:, torch.argmin(sum_maps[0, :]), :, :]
        return {
            "heatmap": heatmap,
            "confidence": conf,
            "detection": det,
            "noiseprint": npp,
        }

    @classmethod
    def from_config(
        cls, config: Optional[str | Dict[str, Any]], device: Optional[str] = "cpu"
    ):
        if isinstance(config, str):
            config = load_yaml(config)

        if config is None:
            trufor_config = TruForConfig()
        else:
            trufor_config = TruForConfig(**config)

        # FIXME typing issue with device. Why is it optional in from config?
        return cls(cfg=trufor_config, device=device)

    def method_to_device(self, device: str):
        self.to(device)
        self.device = torch.device(device)
