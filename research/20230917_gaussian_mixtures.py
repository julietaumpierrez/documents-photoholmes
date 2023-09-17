# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture

# %%
data1 = np.random.normal(-3, 1, (100, 2))
data2 = np.random.normal(3, 0.5, (100, 2))
data = np.vstack((data1, data2))

# %%
plt.scatter(data[:, 0], data[:, 1])
# %%
gm = GaussianMixture(n_components=2)
mus, covs = gm.fit(data)
print(mus)
print(covs)

# %%
