# code extracted from https://github.com/mjkwon2021/CAT-Net/blob/f1716b0849eb4d94687a02c25bf97229b495bf9e/lib/models/network_CAT.py#L286
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
Aug 22, 2020
"""
import random
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from photoholmes.methods.base import BaseTorchMethod
from photoholmes.methods.psccnet.config import PSCCArchConfig, pretrained_arch
from photoholmes.methods.psccnet.network.detection_head import DetectionHead
from photoholmes.methods.psccnet.network.NLCDetection import NLCDetection
from photoholmes.methods.psccnet.network.seg_hrnet import HighResolutionNet
from photoholmes.methods.psccnet.utils import load_network_weight


class PSCCNet(BaseTorchMethod):
    def __init__(
        self,
        weights_paths: Dict[str, str],
        arch_config: Union[PSCCArchConfig, Literal["pretrained"]] = "pretrained",
        device: str = "cuda:0",
        device_ids: Optional[List] = None,
        crop_size: List[int] = [256, 256],
        seed: int = 42,
        **kwargs,
    ):
        """
        weights_paths = {
        """
        random.seed(seed)
        super().__init__(**kwargs)

        self.device = device
        self.device_ids = device_ids

        if arch_config == "pretrained":
            arch_config = pretrained_arch

        FENet = HighResolutionNet(pretrained_arch, **kwargs)
        FENet.init_weights(weights_paths["pretrained"], device)
        SegNet = NLCDetection(arch_config, crop_size)
        ClsNet = DetectionHead(arch_config, crop_size)

        FENet = self.init_network(FENet, weights_paths["FENet"])
        SegNet = self.init_network(SegNet, weights_paths["SegNet"])
        ClsNet = self.init_network(ClsNet, weights_paths["ClsNet"])

        self.FENet = FENet
        self.SegNet = SegNet
        self.ClsNet = ClsNet

    def init_network(self, net, weights_path):
        net = net.to(self.device)
        net = nn.DataParallel(net, device_ids=self.device_ids)
        load_network_weight(net, weights_path, self.device)
        return net.module.to(self.device)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.FENet.eval()
        feat = self.FENet(image)

        # localization head
        self.SegNet.eval()
        heatmap = self.SegNet(feat)[0]

        heatmap = F.interpolate(
            heatmap,
            size=(image.size(2), image.size(3)),
            mode="bilinear",
            align_corners=True,
        )

        # classification head
        self.ClsNet.eval()
        pred_logit = self.ClsNet(feat)
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)[:, 1]

        return {"heatmap": heatmap, "score": pred_logit}
