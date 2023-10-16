# %%
import os
from dataclasses import dataclass
from re import I
from typing import Dict, List

import jpegio
import numpy as np
import torch
from cv2 import dct
from matplotlib.pyplot import sca
from numpy.typing import NDArray
from PIL import Image

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")


# %%
NUM_CHANNELS = 1
# image_path = "data/IMD2020/1a07yi/c8swtoq_0.jpg"
image_path = "data/example_input.jpg"
img = Image.open(image_path)

# %%
jpeg_img = jpegio.read(image_path)
comp_info = jpeg_img.comp_info
scaling = [[comp_info[i].v_samp_factor, comp_info[i].h_samp_factor] for i in range(3)]
if (
    comp_info[0].v_samp_factor
    == comp_info[1].v_samp_factor
    == comp_info[2].v_samp_factor
):
    scaling[0][0] = scaling[1][0] = scaling[2][0] = 2
if (
    comp_info[0].h_samp_factor
    == comp_info[1].h_samp_factor
    == comp_info[2].h_samp_factor
):
    scaling[0][1] = scaling[1][1] = scaling[2][1] = 2


# %%
def upscale_dct_coeffs(dct_coef_array: NDArray, scaling: List[List[int]]) -> NDArray:
    """
    Upscale the DCT coefficients to the original image size.
    """
    r, c = dct_coef_array.shape
    coef_view = dct_coef_array.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
    if scaling[0] == 1 and scaling[1] == 1:
        out_arr = np.zeros((r * 2, c * 2))
        out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
        out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
        out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
        out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
        out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

    elif scaling[0] == 1 and scaling[1] == 2:
        out_arr = np.zeros((r * 2, c))
        out_view = out_arr.reshape(r * 2 // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        out_view[::2, :, :, :] = coef_view[:, :, :, :]
        out_view[1::2, :, :, :] = coef_view[:, :, :, :]

    elif scaling[0] == 2 and scaling[1] == 1:
        out_arr = np.zeros((r, c * 2))
        out_view = out_arr.reshape(r * 2 // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        out_view[:, ::2, :, :] = coef_view[:, :, :, :]
        out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

    elif scaling[0] == 2 and scaling[1] == 2:
        out_arr = np.zeros((r, c))
        out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        out_view[:, :, :, :] = coef_view[:, :, :, :]
    else:
        raise ValueError(
            f"JPEG scaling isn't valid: h_samp_factor = {r}, v_samp_factor = {c}"
        )

    return out_arr


# %%
upscale_dct_coeffs(jpeg_img.coef_arrays[0], scaling[0])
upscale_dct_coeffs(jpeg_img.coef_arrays[1], scaling[1])
upscale_dct_coeffs(jpeg_img.coef_arrays[2], scaling[2])


# %%
@dataclass
class JPEGImage:
    image: NDArray
    dct_coeffs: List[NDArray]
    qtable: List[NDArray]

    def to_dict(self) -> Dict[str, NDArray]:
        return self.__dict__

    @classmethod
    def load(cls, img_path: str, n_dct_channels: int = 1):
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        jpeg = jpegio.read(img_path)
        comp_info = jpeg.comp_info

        scaling = [
            [comp_info[i].v_samp_factor, comp_info[i].h_samp_factor]
            for i in range(n_dct_channels)
        ]
        dct_coeffs = [
            upscale_dct_coeffs(jpeg.coef_arrays[i], scaling[i])
            for i in range(n_dct_channels)
        ]

        qtables = [
            jpeg.quant_tables[comp_info[i].quant_tbl_no].astype(np.float32)
            for i in range(n_dct_channels)
        ]

        return cls(img, dct_coeffs, qtables)


# %%
img = JPEGImage.load(image_path)


# %%
import random


def get_binary_volume(x: torch.Tensor, T: int = 20) -> torch.Tensor:
    x_vol = torch.zeros(size=(T + 1, x.shape[1], x.shape[2]))
    x_vol[0] += (x == 0).float().squeeze()
    for i in range(1, T):
        x_vol[i] += (x == i).float().squeeze()
        x_vol[i] += (x == -i).float().squeeze()
    x_vol[T] += (x >= T).float().squeeze()
    x_vol[T] += (x <= -T).float().squeeze()

    return x_vol


def prepare_for_catnet(jpeg_img: JPEGImage, n_dct_channles: int = 1):
    img = torch.tensor(jpeg_img.image).permute(2, 0, 1)
    dct_coeffs = jpeg_img.dct_coeffs

    h, w = img.shape[1:]

    crop_size = ((h // 8) * 8, (w // 8) * 8)
    if h < crop_size[0] or w < crop_size[1]:
        temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
        temp[: img.shape[0], : img.shape[1], :] = img
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

    s_r = (random.randint(0, max(h - crop_size[0], 0)) // 8) * 8
    s_c = (random.randint(0, max(h - crop_size[1], 0)) // 8) * 8
    img = img[s_r : s_r + crop_size[0], s_c : s_c + crop_size[1], :]
    for i in range(n_dct_channles):
        dct_coeffs[i] = dct_coeffs[i][
            s_r : s_r + crop_size[0], s_c : s_c + crop_size[1]
        ]
    dct_coeffs = torch.tensor(dct_coeffs, dtype=torch.float)
    dct_vols = get_binary_volume(dct_coeffs, T=20)

    qtables = np.array(jpeg_img.qtable[:n_dct_channles])

    return torch.concatenate((img, dct_vols)), torch.tensor(qtables, dtype=torch.float)


# %%j
x, qtable = prepare_for_catnet(img)


# %%
import yaml

from photoholmes.models.catnet.model import CAT_Net

config = yaml.load(open("weights/catnet.yaml", "r"), Loader=yaml.FullLoader)
model = CAT_Net(config["MODEL"]["EXTRA"], num_classes=config["DATASET"]["NUM_CLASSES"])
weights = torch.load("weights/CAT_full_v2.pth.tar", map_location="cpu")
model.load_state_dict(weights["state_dict"])
model.eval()

# %%
import matplotlib.pyplot as plt

plt.imshow(img.image)
# %%
with torch.no_grad():
    out = model(x[None, :], qtable[None, :])

# %%
pred = out.squeeze(0)
labels = torch.nn.functional.softmax(pred, dim=0)[1]
# %%
labels.shape

# %%
plt.imshow(labels.numpy())

# %%
