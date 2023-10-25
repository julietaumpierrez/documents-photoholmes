# %%
import os

from sympy import plot

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from photoholmes.data.input import ColumbiaDataset
from photoholmes.models.splicebuster import Splicebuster
from photoholmes.utils.image import plot_multiple

# %%
COLUMBIA_PATH = "/home/dsense/extra/tesis/datos/columbia"
dataset = ColumbiaDataset(
    COLUMBIA_PATH,
    tampered_only=True,
)
# %%
ims = []
mks = []
for n in range(12):
    im, mk = dataset[n]
    ims.append(im)
    mks.append(mk)

plot_multiple(ims, title="Imágenes Columbia")
plot_multiple(mks, title="Máscaras Columbia")

# %%
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,  # must be 1 to handle arbitrary input sizes
    shuffle=False,  # must be False to get accurate filename
    num_workers=1,
    pin_memory=False,
)
ims = []
mks = []
for im, mk in list(loader)[:12]:
    ims.append(im.squeeze())
    mks.append(mk.squeeze())

plot_multiple(ims, title="Imágenes Columbia")
plot_multiple(mks, title="Máscaras Columbia")
