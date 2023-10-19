# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from photoholmes.models.splicebuster import Splicebuster
from photoholmes.utils.clustering.gaussian_uniform import GaussianUniformEM
from photoholmes.utils.pca import PCA

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
image = Image.open("data/img00.png").convert("L")
np_image = np.array(image) / 255

# %%
model = Splicebuster(mixture="uniform", weights=None)
# %%
features, coords = model.compute_features(np_image)
flat_features = features.reshape(-1, features.shape[-1])

pca = PCA(n_components=model.pca_dim)
flat_features = pca.fit_transform(flat_features)
# %%
gmm = GaussianUniformEM()
gmm.fit(flat_features)
_, labels = gmm.predict(flat_features)

# %%
labels = labels.reshape(features.shape[:2])
# %%
from scipy.interpolate import RegularGridInterpolator

interpolator = RegularGridInterpolator(
    points=coords, values=labels, method="nearest", bounds_error=False, fill_value=0
)

# %%
orig_image_coords = (
    np.array(np.meshgrid(np.arange(np_image.shape[0]), np.arange(np_image.shape[1])))
    .reshape(2, -1)
    .T
)

# %%
mask = interpolator(orig_image_coords).reshape(np_image.shape[:2][::-1]).T
plt.imshow(mask)

# %%
from photoholmes.utils.postprocessing.resizing import upscale_mask

# %%
mask = upscale_mask(coords, labels, np_image.shape[:2], method="nearest")
plt.imshow(mask)
