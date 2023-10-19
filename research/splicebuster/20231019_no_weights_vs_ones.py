# %%
import os

import numpy as np
from PIL import Image

from photoholmes.models.splicebuster import Splicebuster

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
sp = Splicebuster.from_config("src/photoholmes/models/splicebuster/config.yaml")

# %%
image = Image.open("data/img00.png").convert("L")
np_image = np.array(image) / 255
# %%
qhh, qhv, qvh, qvv = sp.filter_and_encode(np_image)

# %%
mask = np.ones(np_image.shape, dtype=np.uint8)
# %%
stride = 8
H, W = qhh.shape
x_range = np.arange(0, H - stride + 1, stride)
y_range = np.arange(0, W - stride + 1, stride)

n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
bins = np.arange(0, n_bins + 1)
feat_dim = int(2 * n_bins)
features = np.zeros((len(x_range), len(y_range), feat_dim))
features_w = np.zeros((len(x_range), len(y_range), feat_dim))

for x_i, i in enumerate(x_range):
    for x_j, j in enumerate(y_range):
        block_weights = mask[i : i + stride, j : j + stride]

        Hhh_w = np.histogram(
            qhh[i : i + stride, j : j + stride],
            bins=bins,
            weights=block_weights,
        )[0].astype(float)
        Hhh = np.histogram(
            qhh[i : i + stride, j : j + stride],
            bins=bins,
        )[
            0
        ].astype(float)

        Hvv_w = np.histogram(
            qhv[i : i + stride, j : j + stride],
            bins=bins,
            weights=block_weights,
        )[0].astype(float)
        Hvv = np.histogram(
            qhv[i : i + stride, j : j + stride],
            bins=bins,
        )[
            0
        ].astype(float)
        Hhv_w = np.histogram(
            qvh[i : i + stride, j : j + stride],
            bins=bins,
            weights=block_weights,
        )[0].astype(float)
        Hhv = np.histogram(
            qvh[i : i + stride, j : j + stride],
            bins=bins,
        )[
            0
        ].astype(float)
        Hvh_w = np.histogram(
            qvh[i : i + stride, j : j + stride],
            bins=bins,
        )[
            0
        ].astype(float)

        Hvh = np.histogram(
            qvh[i : i + stride, j : j + stride],
            bins=bins,
            weights=block_weights,
        )[0].astype(float)

        assert (Hhh == Hhh_w).all()
        assert (Hhv == Hhv_w).all()
        assert (Hvh == Hvh_w).all()
        assert (Hvv == Hvv_w).all()

        features[x_i, x_j] = np.concatenate((Hhh + Hvv, Hhv + Hvh)) / 2
        features_w[x_i, x_j] = np.concatenate((Hhh_w + Hvv_w, Hhv_w + Hvh_w)) / 2


# %%
(features == features_w).all()

# %%
