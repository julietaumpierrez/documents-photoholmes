# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from photoholmes.models.splicebuster.model import Splicebuster
from photoholmes.utils.PCA.pca import PCA as PCA_class

if "research" in os.path.abspath("."):
    os.chdir("../../")


# %%
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2, whiten=True)
pca.fit_transform(X)
print(pca.components_)

# %%
pca_mine = PCA_class(n_components=2, whiten=True)
pca_mine.fit_transform(X)
print(pca_mine.pca.components_)

# %%
image = Image.open("benchmarking/test_images/images/Im_4.jpg").convert("L")
np_image = np.array(image)

# %%
# sp = Splicebuster(stride=64)

# %%
# heatmap = sp.predict(np_image)
# %%
# plt.imshow(heatmap[0])
# plt.figure()
# plt.imshow(heatmap[1])

# %%
