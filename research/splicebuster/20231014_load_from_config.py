# %%
import os

import numpy as np
from PIL import Image

from photoholmes.models.splicebuster import Splicebuster

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
sp = Splicebuster.from_config("src/photoholmes/models/splicebuster/config.yaml")

# %%
