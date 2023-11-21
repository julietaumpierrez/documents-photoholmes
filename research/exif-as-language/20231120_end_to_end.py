# %%
import os

import matplotlib.pyplot as plt

from photoholmes.models.exif_as_language.method import EXIFAsLanguage
from photoholmes.models.exif_as_language.preprocessing import exif_preprocessing
from photoholmes.utils.image import read_image

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
image = read_image("data/img00.png") / 255
plt.imshow(image.permute(1, 2, 0))

# %%
method = EXIFAsLanguage(
    "distilbert",
    "resnet50",
    device="cpu",
    state_dict_path="weights/exif/pruned_weights.pth",
)
# %%
exif_input = exif_preprocessing(image=image)
out = method.predict(img=exif_input["image"])

# %%
plt.imshow(out["ms"])
plt.show()
# %%
