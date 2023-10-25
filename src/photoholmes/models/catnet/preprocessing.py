from random import randint
from typing import Dict

import torch
from torch import Tensor

from photoholmes.utils.preprocessing.base import BaseTransform, PreProcessingPipeline
from photoholmes.utils.preprocessing.image import ToTensor


def get_binary_volume(x: Tensor, T: int = 20) -> Tensor:
    x_vol = torch.zeros(size=(T + 1, x.shape[1], x.shape[2]))

    x_vol[0] += (x == 0).float().squeeze()

    indices = torch.arange(1, T).unsqueeze(1).unsqueeze(2)
    x_vol[1:T] += ((x == indices) | (x == -indices)).float().squeeze()

    x_vol[T] += (x >= T).float().squeeze()
    x_vol[T] += (x <= -T).float().squeeze()

    return x_vol


class CatnetPreprocessing(BaseTransform):
    def __init__(self, n_dct_channels: int = 1, T: int = 20):
        self.n_dct_channels = n_dct_channels
        self.T = T

    def __call__(
        self, image: Tensor, dct_coefficients: Tensor, qtables: Tensor
    ) -> Dict[str, Tensor]:
        h, w = image.shape[1:]

        crop_size = ((h // 8) * 8, (w // 8) * 8)
        if h < crop_size[0] or w < crop_size[1]:
            temp = torch.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
            temp[: image.shape[0], : image.shape[1], :] = image
            max_h = max(
                crop_size[0],
                max([dct_coefficients[c].shape[0] for c in range(self.n_dct_channels)]),
            )
            max_w = max(
                crop_size[1],
                max([dct_coefficients[c].shape[1] for c in range(self.n_dct_channels)]),
            )
            for i in range(self.n_dct_channels):
                temp = torch.full((max_h, max_w), 0.0)  # pad with 0
                temp[
                    : dct_coefficients[i].shape[0], : dct_coefficients[i].shape[1]
                ] = dct_coefficients[i][:, :]
                dct_coefficients[i] = temp

        s_r = (randint(0, max(h - crop_size[0], 0)) // 8) * 8
        s_c = (randint(0, max(w - crop_size[1], 0)) // 8) * 8
        image = image[:, s_r : s_r + crop_size[0], s_c : s_c + crop_size[1]]
        for i in range(self.n_dct_channels):
            dct_coefficients[i] = dct_coefficients[i][
                s_r : s_r + crop_size[0], s_c : s_c + crop_size[1]
            ]
        t_dct_coeffs = torch.tensor(dct_coefficients, dtype=torch.float)

        image = (image - 127.5) / 127.5
        t_dct_vols = get_binary_volume(t_dct_coeffs, T=self.T)

        qtables = torch.tensor(qtables, dtype=torch.int16)

        x = torch.concatenate((image, t_dct_vols))
        return {"x": x, "qtable": qtables}


catnet_preprocessing = PreProcessingPipeline(
    transforms=[ToTensor(), CatnetPreprocessing()]
)
