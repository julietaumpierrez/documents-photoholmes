# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.methods.dq import DQ, dq_preprocessing

# %%
method = DQ.from_config("src/photoholmes/methods/dq/config.yaml")
# %%
from photoholmes.utils.image import read_image, read_jpeg_data

# %%
image = read_image("data/pelican.png")
dct_coefficients, _ = read_jpeg_data("data/pelican.png", num_dct_channels=1)

# %%
image_data = {"image": image, "dct_coefficients": dct_coefficients}
# %%
# %%
input = dq_preprocessing(**image_data)
input
# %%
output_1 = method.predict(**input)
output_1
# %%
from photoholmes.utils.image import plot

# %%
plot(output_1)
# %%
output_2 = method.benchmark(**input)
output_2
# %%
