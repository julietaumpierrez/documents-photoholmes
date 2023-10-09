# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture  # type: ignore

from photoholmes.utils.clustering.gaussian_uniform import GaussianUniformEM as Gum_new
from photoholmes.utils.clustering.gum import GaussianUniformMixture as Gum_orig

# %%
data1 = np.random.normal(-3, 1, (100, 2))
data2 = np.random.uniform(3, 0.5, (100, 2))
data = np.hstack((data1, data2))

# %%
plt.scatter(data[:, 0], data[:, 1])
# %%
gum_orig = Gum_orig()
gum_orig.fit(data)
mahalanobis_orig, gammas_orig = gum_orig.predict(data)

gum_new = Gum_new()
gum_new.fit(data)
mahalanobis_new, gammas_new = gum_new.predict(data)

print(mahalanobis_new == mahalanobis_orig)
# %%
