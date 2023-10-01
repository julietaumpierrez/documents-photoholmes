# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
image = Image.open("data/img00.png").convert("L")
np_image = np.array(image) / 255
plt.imshow(np_image, cmap="gray")
plt.show()

# %%
BLOCK_SIZE = 128
STRIDE = 8
T = 1
q = 2


# %%
def third_order_residual(x: np.ndarray, axis: int = 0):
    if axis == 0:
        residual = x[:, :-3] - 3 * x[:, 1:-2] + 3 * x[:, 2:-1] - x[:, 3:]
        residual = residual[2:-2, 1:]
    elif axis == 1:
        residual = x[:-3] - 3 * x[1:-2] + 3 * x[2:-1] - x[3:]
        residual = residual[1:, 2:-2]
    else:
        raise ValueError("axis must be 0 (horizontal) or 1 (vertical)")
    return residual


h_residual = third_order_residual(np_image, axis=0)
v_residual = third_order_residual(np_image, axis=1)


# %%
def qround(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.floor(np.abs(y) + 0.5)


def quantize(x: np.ndarray, T=1, q=2) -> np.ndarray:
    """
    Uniform quantization used in the paper.
    """
    q = 3 * float(q) / 256
    if isinstance(x, np.ndarray):
        return np.clip(qround(x / q) + T, 0, 2 * T)


def quantize_marina(x):
    th = np.float32(6 / 256)
    bin2 = 2 * (x > th / 2)
    bin1 = (x <= th / 2) * (x >= -th / 2)
    return bin2 + bin1


qh_residual = quantize(h_residual)
qh_residual_marina = quantize_marina(h_residual)
assert (qh_residual == qh_residual_marina).all()
qv_residual = quantize(v_residual)
qv_residual_marina = quantize_marina(v_residual)
assert (qv_residual == qv_residual_marina).all()

plt.imshow(qh_residual, cmap="gray")
plt.figure()
plt.imshow(qv_residual, cmap="gray")


# %%
def encode_matrix(X: np.ndarray, T: int = 1, k: int = 4, axis=0) -> np.ndarray:
    coded_shape = (X.shape[0] - k + 1, X.shape[1] - k + 1)

    encoded_matrix = np.zeros((2, *coded_shape))
    base = 2 * T + 1
    max_value = (2 * T + 1) ** k - 1

    if axis == 0:
        for i, b in enumerate(range(k)):
            encoded_matrix[0] += base**b * X[: -k + 1, i : coded_shape[1] + i]
            encoded_matrix[1] += (
                base ** (k - 1 - b) * X[: -k + 1, i : coded_shape[1] + i]
            )

    elif axis == 1:
        for i, b in enumerate(range(k)):
            encoded_matrix[0] += base**b * X[i : coded_shape[0] + i, : -k + 1]
            encoded_matrix[1] += (
                base ** (k - 1 - b) * X[i : coded_shape[0] + i, : -k + 1]
            )

    reduced = np.minimum(encoded_matrix[0], encoded_matrix[1])
    reduced = np.minimum(reduced, max_value - encoded_matrix[0])
    reduced = np.minimum(reduced, max_value - encoded_matrix[1])

    _, reduced = np.unique(reduced, return_inverse=True)
    return reduced.reshape(coded_shape)


import time

t0 = time.time()
qhh = encode_matrix(qh_residual)
qhv = encode_matrix(qh_residual, axis=1)
qvh = encode_matrix(qv_residual)
qvv = encode_matrix(qv_residual, axis=1)
t1 = time.time()

print(f"Coding { t1 - t0 }s")
# %%
sample = qhh[:BLOCK_SIZE, :BLOCK_SIZE]
Hhh, _ = np.histogram(sample, density=True, bins=np.arange(25))
# %%
from tqdm import tqdm

x_range = np.arange(0, np_image.shape[0] - STRIDE + 1, STRIDE)
y_range = np.arange(0, np_image.shape[1] - STRIDE + 1, STRIDE)
pbar = tqdm(total=x_range.shape[0] * y_range.shape[0])
n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
features = np.zeros((x_range.shape[0], y_range.shape[0], 2 * n_bins))
for x_i, i in enumerate(x_range):
    for x_j, j in enumerate(y_range):
        Hhh = np.histogram(qhh[i : i + STRIDE, j : j + STRIDE], bins=n_bins)[0].astype(
            float
        )
        Hvv = np.histogram(qhv[i : i + STRIDE, j : j + STRIDE], bins=n_bins)[0].astype(
            float
        )
        Hhv = np.histogram(qvh[i : i + STRIDE, j : j + STRIDE], bins=n_bins)[0].astype(
            float
        )
        Hvh = np.histogram(qvv[i : i + STRIDE, j : j + STRIDE], bins=n_bins)[0].astype(
            float
        )

        # Hhh /= np.sum(Hhh)
        # Hvv /= np.sum(Hvv)
        # Hhv /= np.sum(Hhv)
        # Hvh /= np.sum(Hvh)

        features[x_i, x_j] = np.concatenate((Hhh + Hvv, Hhv + Hvh)) / 2
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
flat_features = block_features.reshape(-1, block_features.shape[-1])
# block_features = [f for f in block_features]
# %%
from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture

gmm = GaussianMixture()

# %%
mus, covs = gmm.fit(flat_features)
# %%
from photoholmes.models.splicebuster.utils import mahalanobis_distance

# %%
x_cent = flat_features - mus[0]
x_cent[0] @ np.linalg.inv(covs[0]) @ x_cent[0].T
# %%
labels = mahalanobis_distance(block_features, mus[1], covs[1]) / mahalanobis_distance(
    block_features, mus[0], covs[0]
)
labels_comp = 1 / labels
# %%
heatmap = np.zeros((2, np_image.shape[0] - BLOCK_SIZE, np_image.shape[1] - BLOCK_SIZE))
k = 0
for i in range(0, features.shape[0] - strides_x_block):
    for j in range(0, features.shape[1] - strides_x_block):
        heatmap[0][
            STRIDE * i : STRIDE * (i + 1), STRIDE * j : STRIDE * (j + 1)
        ] = labels[k]
        heatmap[1][
            STRIDE * i : STRIDE * (i + 1), STRIDE * j : STRIDE * (j + 1)
        ] = labels_comp[k]
        k += 1
# %%
plt.figure()
plt.imshow(heatmap[0], cmap="gray")
plt.figure()
plt.imshow(heatmap[1], cmap="gray")
# %%
pred = heatmap[heatmap.sum(axis=-1).sum(axis=-1).argmin()]
pred /= pred.max()
plt.imshow(pred)
# %%
labels.sum(), labels_comp.sum()

# %%
t_labels = labels if labels.sum() < labels_comp.sum() else labels_comp
# %%
heatmap = np.zeros((np_image.shape[0], np_image.shape[1]))
n_label = 0
for i in range(0, np_image.shape[0] - BLOCK_SIZE, STRIDE):
    for j in range(0, np_image.shape[1] - BLOCK_SIZE, STRIDE):
        heatmap[i : i + STRIDE, j : j + STRIDE] = t_labels[n_label]
        n_label += 1

# %%
