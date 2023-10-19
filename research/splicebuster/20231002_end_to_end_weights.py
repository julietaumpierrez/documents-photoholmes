# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from photoholmes.models.splicebuster.utils import (
    encode_matrix,
    get_saturated_region_mask,
    quantize,
    third_order_residual,
)

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")

# %%
img = np.array(Image.open("data/img00.png")) / 255
plt.imshow(img)

mask = get_saturated_region_mask(img)
plt.imshow(mask)

img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

# %%
BLOCK_SIZE = 128
STRIDE = 8
T = 1
q = 2

# %%
qh_residual = quantize(third_order_residual(img, axis=0), T, q)
qv_residual = quantize(third_order_residual(img, axis=1), T, q)
# %%
qhh = encode_matrix(qh_residual)
qhv = encode_matrix(qh_residual, axis=1)
qvh = encode_matrix(qv_residual)
qvv = encode_matrix(qv_residual, axis=1)
# %%
mask = mask[2:-5, 2:-5]
# %%
x_range = np.arange(0, img.shape[0] - STRIDE + 1, STRIDE)
y_range = np.arange(0, img.shape[1] - STRIDE + 1, STRIDE)
pbar = tqdm(total=x_range.shape[0] * y_range.shape[0])
n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
features = np.zeros((x_range.shape[0], y_range.shape[0], 2 * n_bins))
features_w = np.zeros((x_range.shape[0], y_range.shape[0], 2 * n_bins))
window_weights = np.ones((x_range.shape[0], y_range.shape[0]))
for x_i, i in enumerate(x_range):
    for x_j, j in enumerate(y_range):
        weights = mask[i : i + STRIDE, j : j + STRIDE]
        window_weights[x_i, x_j] = weights.sum() / STRIDE**2

        Hhh_w = np.histogram(
            qhh[i : i + STRIDE, j : j + STRIDE], bins=n_bins, weights=weights
        )[0].astype(float)
        Hvv_w = np.histogram(
            qhv[i : i + STRIDE, j : j + STRIDE], bins=n_bins, weights=weights
        )[0].astype(float)
        Hhv_w = np.histogram(
            qvh[i : i + STRIDE, j : j + STRIDE], bins=n_bins, weights=weights
        )[0].astype(float)
        Hvh_w = np.histogram(
            qvv[i : i + STRIDE, j : j + STRIDE], bins=n_bins, weights=weights
        )[0].astype(float)

        features[x_i, x_j] = np.concatenate((Hhh + Hvv, Hhv + Hvh)) / 2
        features_w[x_i, x_j] = np.concatenate((Hhh_w + Hvv_w, Hhv_w + Hvh_w)) / 2
        pbar.update(1)
pbar.close()
# %%
strides_x_block = BLOCK_SIZE // STRIDE
block_features = np.zeros(
    (
        features.shape[0] - strides_x_block,
        features.shape[1] - strides_x_block,
        2 * n_bins,
    )
)
for i in range(block_features.shape[0]):
    for j in range(block_features.shape[1]):
        block_features[i, j] = features[
            i : i + strides_x_block, j : j + strides_x_block
        ].sum(axis=(0, 1))
        block_features[i, j] /= np.sum(block_features[i, j])
flat_features = np.sqrt(block_features.reshape(-1, block_features.shape[-1]))

# %%
from photoholmes.models.splicebuster.utils import mahalanobis_distance
from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture
from photoholmes.utils.pca import PCA

gmm = GaussianMixture()
pca = PCA()
# %%
flat_features = pca.fit_transform(flat_features)
mus, covs = gmm.fit(flat_features)

labels = mahalanobis_distance(flat_features, mus[0], covs[0]) / mahalanobis_distance(
    flat_features, mus[1], covs[1]
)
labels_comp = 1 / labels
labels = labels_comp if labels_comp.sum() < labels.sum() else labels
labels = labels.reshape(block_features.shape[:2])
# %%
plt.imshow(labels)
print(labels.shape)
