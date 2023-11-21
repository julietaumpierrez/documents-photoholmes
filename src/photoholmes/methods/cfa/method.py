# code extracted from https://github.com/qbammey/adaptive_cfa_forensics/blob/master/src/structure.py
# ------------------------------------------------------------------------------
# Written by Quentin Bammey (quentin.bammey@ens-paris-saclay.fr)
# ------------------------------------------------------------------------------

"""Internal structures used in the network."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from photoholmes.methods.base import BaseTorchMethod
from photoholmes.methods.cfa.config import CFAConfig
from photoholmes.utils.generic import load_yaml

logger = logging.getLogger(__name__)


class DirFullDil(nn.Module):
    """
    Performs horizontal, vertical and full convolutions, concatenate them, then perform
    the same number of horizontal, vertical and full convolutions.
    In parallel, performs horizontal, vertical and full convolutions once, but with a
    dilation factor of 2.
    Returns the concatenated results number of parametres:
    (2*n_dir + n_full)*(1 + 2*n_dir + n_full + channels_in) +
    (channels_in + 1)*(2*n_dir_dil + n_full_dil)
    """

    def __init__(self, channels_in, *n_convolutions):
        super(DirFullDil, self).__init__()
        n_dir, n_full, n_dir_dil, n_full_dil = n_convolutions
        self.h1 = nn.Conv2d(channels_in, n_dir, (1, 3))
        self.h2 = nn.Conv2d(2 * n_dir + n_full, n_dir, (1, 3))
        self.v1 = nn.Conv2d(channels_in, n_dir, (3, 1))
        self.v2 = nn.Conv2d(2 * n_dir + n_full, n_dir, (3, 1))
        self.f1 = nn.Conv2d(channels_in, n_full, 3)
        self.f2 = nn.Conv2d(2 * n_dir + n_full, n_full, 3)
        self.hd = nn.Conv2d(channels_in, n_dir_dil, (1, 3), dilation=2)
        self.vd = nn.Conv2d(channels_in, n_dir_dil, (3, 1), dilation=2)
        self.fd = nn.Conv2d(channels_in, n_full_dil, 3, dilation=2)
        self.channels_out = 2 * n_dir + n_full + 2 * n_dir_dil + n_full_dil

    def forward(self, x):
        h_d = self.hd(x)[:, :, 2:-2]
        v_d = self.vd(x)[:, :, :, 2:-2]
        f_d = self.fd(x)
        h = self.h1(x)[:, :, 1:-1]
        v = self.v1(x)[:, :, :, 1:-1]
        f = self.f1(x)
        x = F.softplus(torch.cat((h, v, f), 1))
        h = self.h2(x)[:, :, 1:-1]
        v = self.v2(x)[:, :, :, 1:-1]
        f = self.f2(x)
        return torch.cat((h_d, v_d, f_d, h, v, f), 1)


class SkipDoubleDirFullDir(nn.Module):
    """Uses a first DirFullDir module, skips the input to the results of the first
    module, and finally send everything through a second DirFullDir module."""

    def __init__(self, channels_in, convolutions_1, convolutions_2):
        super(SkipDoubleDirFullDir, self).__init__()
        self.conv1 = DirFullDil(3, *convolutions_1)
        self.conv2 = DirFullDil(3 + self.conv1.channels_out, *convolutions_2)
        self.channels_out = self.conv2.channels_out
        self.padding = 4

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat((x[:, :, 2:-2, 2:-2], x1), 1)
        x2 = self.conv2(x1)
        x2 = torch.cat((x1[:, :, 2:-2, 2:-2], x2), 1)
        return x2


class SeparateAndPermutate(nn.Module):
    def forward(self, x):
        n, C, Y, X = x.shape
        assert n == 1
        assert Y % 2 == 0
        assert X % 2 == 0
        x_00 = x[:, :, ::2, ::2]
        x_01 = x[:, :, ::2, 1::2]
        x_10 = x[:, :, 1::2, ::2]
        x_11 = x[:, :, 1::2, 1::2]

        ind = [k + C * i for k in range(C) for i in range(4)]

        xx_00 = torch.cat((x_00, x_01, x_10, x_11), 1)[:, ind]
        xx_01 = torch.cat((x_01, x_00, x_11, x_10), 1)[:, ind]
        xx_10 = torch.cat((x_10, x_11, x_00, x_01), 1)[:, ind]
        xx_11 = torch.cat((x_11, x_10, x_01, x_00), 1)[:, ind]

        x = torch.cat((xx_00, xx_01, xx_10, xx_11), 0)
        return x


class Pixelwise(nn.Module):
    def __init__(self):
        super(Pixelwise, self).__init__()
        self.conv1 = nn.Conv2d(103, 30, 1)
        self.conv2 = nn.Conv2d(30, 15, 1)
        self.conv3 = nn.Conv2d(45, 15, 1)
        self.conv4 = nn.Conv2d(60, 30, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x2), 1)))
        x4 = self.conv4(torch.cat((x, x2, x3), 1))
        return x4


class CFANet(BaseTorchMethod):
    """
    Input image: (1, 3, 2Y, 2X)
    ⮟First module: Spatial convolutions, without pooling
    Pixel-wise features: (1, 30 , 2Y-8, 2X-8) (⮞Pixelwise auxiliary training)
    ⮟Grid separation and permutations
    : (1, 120, Y-4, X-4)
    ⮟Permutations of the grid pixels
    Pixel-wise features, with the pixels separated by position in the grid in different
    channels, permutated in the four possible ways in image numbers: (4, 120, Y-4, X-4)
    ⮟ 1×1 convolutions: pixel-wise causality
    ⮟Average Pooling to get the mean in each block
    Mean in each block, channel and permutation of the pixels' features:
    (4N, 120, (Y-4)//block_size, (X-4)//block_size
    ⮟1×1 convolutions: block-wise causality
    Features in each block, channel and permutation of the pixels's features:
    (4N, 4, (Y-4)//block_size, (X-4)//block_size)
    ⮟LogSoftMax
    Out
    """

    def __init__(
        self,
        weights: Optional[Union[str, Path, dict]] = None,
        **kwargs,
    ):
        """
        :param spatial: nn.Module, must have spatial.padding defined
        :param causal: CausalNet instance
        output channels of spatial must match with input channels of causal
        """
        super().__init__(**kwargs)
        self.spatial = SkipDoubleDirFullDir(3, (10, 5, 10, 5), (10, 5, 10, 5))
        self.pixelwise = Pixelwise()
        self.blockwise = nn.Sequential(
            nn.Conv2d(120, 180, 1, groups=30),
            nn.Softplus(),
            nn.Conv2d(180, 90, 1, groups=30),
            nn.Softplus(),
            nn.Conv2d(90, 90, 1, groups=30),
            nn.Softplus(),
            nn.Conv2d(90, 45, 1),
            nn.Softplus(),
            nn.Conv2d(45, 45, 1),
            nn.Softplus(),
            nn.Conv2d(45, 4, 1),
            nn.LogSoftmax(dim=1),
        )
        self.pixelwise = Pixelwise()
        self.auxiliary = nn.Sequential(
            self.spatial, self.pixelwise, nn.Conv2d(30, 4, 1), nn.LogSoftmax(dim=1)
        )
        self.grids = SeparateAndPermutate()
        self.padding = self.spatial.padding

        if weights is not None:
            self.load_weigths(weights)
        else:
            self.init_weights()

    def init_weights(self):
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, block_size=32):
        x = self.spatial(x)
        x = self.pixelwise(x)
        x = self.grids(x)
        x = F.avg_pool2d(x, block_size // 2)
        x = self.blockwise(x)
        return x

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        Y_o, X_o = x.shape[-2:]
        pred = self.forward(x).cpu()
        pred = torch.exp(pred)

        pred[:, 1] = pred[torch.tensor([1, 0, 3, 2]), 1]
        pred[:, 2] = pred[torch.tensor([2, 3, 0, 1]), 2]
        pred[:, 3] = pred[torch.tensor([3, 2, 1, 0]), 3]
        pred = torch.mean(pred, dim=1)

        best_grid = torch.argmax(torch.mean(pred, dim=(1, 2)))
        authentic = torch.argmax(pred, dim=0) == best_grid
        confidence = 1 - torch.max(pred, dim=0).values
        confidence = torch.clamp(confidence, 0, 1)
        confidence[authentic] = 1

        heatmap = confidence.numpy()

        upscaled_heatmap = heatmap.repeat(32, axis=0).repeat(32, axis=1)
        output = np.zeros((Y_o, X_o))
        output[
            : upscaled_heatmap.shape[0], : upscaled_heatmap.shape[1]
        ] = upscaled_heatmap
        return output

    def load_weigths(self, weights: Union[str, Path, dict]):
        if isinstance(weights, (str, Path)):
            weights = torch.load(
                weights, map_location=next(self.parameters())[0].device
            )

        if isinstance(weights, dict) and "state_dict" in weights.keys():
            weights = weights["state_dict"]

        self.load_state_dict(weights)  # type: ignore

    @classmethod
    def from_config(cls, config: Optional[Union[CFAConfig, str, Path, dict]]):
        if isinstance(config, (str, Path)):
            config = load_yaml(str(config))

        if isinstance(config, CFAConfig):
            config = config.__dict__

        if config is None:
            config = {"weights": None}
        return cls(**config)
