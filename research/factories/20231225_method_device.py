# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from photoholmes.methods.method_factory import MethodFactory

# %%
method, preprocessing = MethodFactory.load("exif_as_language", device="cuda")
# %%
method.model_to_device()

# %%
