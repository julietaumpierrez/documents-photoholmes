# %%
import os

if "research" in os.getcwd():
    os.chdir("../..")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
import matplotlib.pyplot as plt

# %%
import numpy as np

from photoholmes.preprocessing.image import RGBtoYCrCb, ToNumpy
from photoholmes.utils.image import read_image

img = ToNumpy()(read_image("data/COVERAGE/image/21t.tif"))["image"]
plt.imshow(img)

# %%
rgb_to_y = RGBtoYCrCb()

ph_ill = rgb_to_y(image=img)["image"][..., 0]
plt.imshow(ph_ill, cmap="gray")

# %%
zr_ill = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

plt.imshow(zr_ill, cmap="gray")
# %%
from photoholmes.methods.zero import Zero
from photoholmes.methods.zero.preprocessing import zero_preprocessing

zero = Zero()

# %%
img = ToNumpy()(read_image("data/COVERAGE/image/21.tif"))["image"]

image = zero_preprocessing(image=img)["image"]

forgery_mask, votes, main_grid = zero.predict(image)
# %%
plt.imshow(forgery_mask, cmap="gray")

# %%
a = ToNumpy()(read_image("../implementations/zero/test/votes_func.tif"))["image"]
plt.imshow(a[..., 0])

# %%
plt.imshow(votes)
# %%
from PIL import Image

img = Image.open("../implementations/zero/test/votes.tif")
votes_orig = np.array(img)
# %%

votes = zero.compute_grid_votes_per_pixel(ph_ill)
# %%
(votes_orig == votes).all()
# %%
diff = votes_orig - votes
print(diff.max())

plt.imshow(np.abs(diff))
# %%
(votes_orig != votes).sum() / votes.size

# %%
# %%
plt.imshow(votes)

# %%
abs(diff).min()

# %%
image_orig = np.array(Image.open("../implementations/zero/test/image.png"))
plt.imshow(image_orig)

# %%
image = zero_preprocessing(image=img)["image"]
plt.imshow(image)
# %%
(image[..., 0] == image_orig).all()
# %%
