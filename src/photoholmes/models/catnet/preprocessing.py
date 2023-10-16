from random import randint
from typing import List, Union

import numpy as np
import torch
from numpy.typing import NDArray


def get_binary_volume(x: torch.Tensor, T: int = 20) -> torch.Tensor:
    x_vol = torch.zeros(size=(T + 1, x.shape[1], x.shape[2]))
    x_vol[0] += (x == 0).float().squeeze()
    for i in range(1, T):
        x_vol[i] += (x == i).float().squeeze()
        x_vol[i] += (x == -i).float().squeeze()
    x_vol[T] += (x >= T).float().squeeze()
    x_vol[T] += (x <= -T).float().squeeze()

    return x_vol


def catnet_preprocessing(
    img: NDArray,
    dct_coeffs: NDArray,
    qtables: Union[List[NDArray], NDArray],
    n_dct_channles: int = 1,
):
    t_img = torch.tensor(img).permute(2, 0, 1)

    h, w = t_img.shape[1:]

    crop_size = ((h // 8) * 8, (w // 8) * 8)
    if h < crop_size[0] or w < crop_size[1]:
        temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
        temp[: t_img.shape[0], : t_img.shape[1], :] = t_img
        max_h = max(
            crop_size[0],
            max([dct_coeffs[c].shape[0] for c in range(n_dct_channles)]),
        )
        max_w = max(
            crop_size[1],
            max([dct_coeffs[c].shape[1] for c in range(n_dct_channles)]),
        )
        for i in range(n_dct_channles):
            temp = np.full((max_h, max_w), 0.0)  # pad with 0
            temp[: dct_coeffs[i].shape[0], : dct_coeffs[i].shape[1]] = dct_coeffs[i][
                :, :
            ]
            dct_coeffs[i] = temp

    s_r = (randint(0, max(h - crop_size[0], 0)) // 8) * 8
    s_c = (randint(0, max(h - crop_size[1], 0)) // 8) * 8
    t_img = t_img[s_r : s_r + crop_size[0], s_c : s_c + crop_size[1], :]
    for i in range(n_dct_channles):
        dct_coeffs[i] = dct_coeffs[i][
            s_r : s_r + crop_size[0], s_c : s_c + crop_size[1]
        ]
    t_dct_coeffs = torch.tensor(dct_coeffs, dtype=torch.float)

    t_img = (t_img - 127.5) / 127.5
    t_dct_vols = get_binary_volume(t_dct_coeffs, T=20)

    qtables = np.array(qtables[:n_dct_channles])

    return torch.concatenate((t_img, t_dct_vols)), torch.tensor(
        qtables, dtype=torch.float
    )
