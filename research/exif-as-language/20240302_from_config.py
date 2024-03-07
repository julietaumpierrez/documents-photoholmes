# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.methods.exif_as_language import EXIFAsLanguage, exif_preprocessing

# %%
method = EXIFAsLanguage.from_config(
    "src/photoholmes/methods/exif_as_language/config.yaml"
)
# %%
from photoholmes.utils.image import read_image

# %%
image = read_image("data/trufor/tampered2.png")[:, :170, :170]
image.shape

# %%
image_data = {"image": image}
# %%
# %%
input = exif_preprocessing(**image_data)
input
# %%
output_1 = method.predict(**input)
output_1
# %%
from photoholmes.utils.image import plot

# %%
plot(output_1[0])
# %%
output_2 = method.benchmark(**input)
output_2
# %%
plot(output_2["heatmap"])
# %%
output_2["heatmap"].max()
# %%
