# code extracted from https://github.com/qbammey/adaptive_cfa_forensics/blob/master/src/structure.py
# ------------------------------------------------------------------------------
# Written by Quentin Bammey (quentin.bammey@ens-paris-saclay.fr)
# ------------------------------------------------------------------------------

"""Internal structures used in the network."""

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from photoholmes.methods.adaptive_cfa_net.config import (
    AdaptiveCFANetArchConfig,
    AdaptiveCFANetConfig,
    pretrained_arch,
)
from photoholmes.methods.base import BaseTorchMethod, BenchmarkOutput
from photoholmes.postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)
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
        self.conv1 = DirFullDil(channels_in, *convolutions_1)
        self.conv2 = DirFullDil(channels_in + self.conv1.channels_out, *convolutions_2)
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
    def __init__(
        self,
        channels_in=103,
        conv1_out_channels=30,
        conv2_out_channels=15,
        conv3_out_channels=15,
        conv4_out_channels=30,
        kernel_size=1,
    ):
        super(Pixelwise, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, conv1_out_channels, kernel_size)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size)
        self.conv3 = nn.Conv2d(
            conv1_out_channels + conv2_out_channels, conv3_out_channels, kernel_size
        )
        self.conv4 = nn.Conv2d(
            conv1_out_channels + conv2_out_channels + conv3_out_channels,
            conv4_out_channels,
            kernel_size,
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x2), 1)))
        x4 = self.conv4(torch.cat((x, x2, x3), 1))
        return x4


class AdaptiveCFANet(BaseTorchMethod):
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
        arch_config: Union[
            AdaptiveCFANetArchConfig, Literal["pretrained"]
        ] = "pretrained",
        weights: Optional[Union[str, Path, dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if arch_config == "pretrained":
            arch_config = pretrained_arch

        self.arch_config = arch_config

        self.load_model(arch_config)

        if weights is not None:
            self.load_weights(weights)
        else:
            self.init_weights()

    def load_model(self, arch_config: AdaptiveCFANetArchConfig):
        # Initialize DirFullDil for the SkipDoubleDirFullDir using config
        conv1_config = arch_config.skip_double_dir_full_dir_config.convolutions_1
        conv2_config = arch_config.skip_double_dir_full_dir_config.convolutions_2
        self.spatial = SkipDoubleDirFullDir(
            arch_config.skip_double_dir_full_dir_config.channels_in,
            (
                conv1_config.n_dir,
                conv1_config.n_full,
                conv1_config.n_dir_dil,
                conv1_config.n_full_dil,
            ),
            (
                conv2_config.n_dir,
                conv2_config.n_full,
                conv2_config.n_dir_dil,
                conv2_config.n_full_dil,
            ),
        )

        # Initialize Pixelwise using config
        pw_config = arch_config.pixelwise_config
        self.pixelwise = Pixelwise(
            pw_config.conv1_in_channels,
            pw_config.conv1_out_channels,
            pw_config.conv2_out_channels,
            pw_config.conv3_out_channels,
            pw_config.conv4_out_channels,
            pw_config.kernel_size,
        )

        # Initialize auxiliary using config
        self.auxiliary = nn.Sequential(
            self.spatial, self.pixelwise, nn.Conv2d(30, 4, 1), nn.LogSoftmax(dim=1)
        )

        # Initialize Blockwise using config
        self.blockwise = nn.Sequential()
        for i, layer in enumerate(arch_config.blockwise_config.layers):
            conv_layer = nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                groups=layer.groups,
            )
            self.blockwise.add_module(f"{2*i}", conv_layer)
            if layer.activation.lower() != "none":
                activation_fn = getattr(nn, layer.activation, None)
                if activation_fn:
                    if activation_fn == nn.LogSoftmax:
                        self.blockwise.add_module(
                            f"activation_{i}", activation_fn(dim=1)
                        )
                    else:
                        self.blockwise.add_module(f"activation_{i}", activation_fn())
                else:
                    raise ValueError(
                        f"Activation function '{layer.activation}' is not supported."
                    )
        self.grids = SeparateAndPermutate()

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
    def predict(self, image: Tensor, image_size: Tuple[int, int]) -> Tensor:

        image = image.to(self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        pred = self.forward(image)
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
        error_map = 1 - confidence

        upscaled_heatmap = simple_upscale_heatmap(error_map, 32)
        upscaled_heatmap = resize_heatmap_with_trim_and_pad(
            upscaled_heatmap, image_size
        )
        return upscaled_heatmap

    def benchmark(self, image: Tensor, image_size: Tuple[int, int]) -> BenchmarkOutput:
        heatmap = self.predict(image, image_size)
        return {"heatmap": heatmap, "mask": None, "detection": None}

    @classmethod
    def from_config(
        cls,
        config: Optional[Union[dict, str, Path, AdaptiveCFANetConfig]],
    ):
        if isinstance(config, AdaptiveCFANetConfig):
            return cls(**config.__dict__)

        if isinstance(config, str) or isinstance(config, Path):
            config = load_yaml(str(config))
        elif config is None:
            config = {}

        adaptive_cga_net_config = AdaptiveCFANetConfig(**config)

        return cls(
            arch_config=adaptive_cga_net_config.arch,
            weights=adaptive_cga_net_config.weights,
        )
