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

from photoholmes.data.input.columbia import ColumbiaDataset
from photoholmes.models.splicebuster import Splicebuster
from photoholmes.utils.image import plot_multiple

# %%
COLUMBIA_PATH = "/home/pento/workspace/fing/photoholmes/data/Columbia_subsample"
dataset = ColumbiaDataset(COLUMBIA_PATH)

ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes Columbia")
plot_multiple(mks, title="Máscaras Columbia")

# %%
COLUMBIA_PATH = "/home/dsense/extra/tesis/datos/columbia"
dataset_jpeg = ColumbiaDataset(COLUMBIA_PATH, item_data=["dct_coefficients"])

ims = []
mks = []

x, mk = dataset_jpeg[0]
print("Stream DCT de primera imagen, canal 0:")
print(x["dct_coefficients"][0])

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
for x, mk in list(loader)[:6]:
    im = x["image"]
    ims.append(im.squeeze())
    mks.append(mk.squeeze())
for x, mk in list(loader)[-6:]:
    im = x["image"]
    ims.append(im.squeeze())
    mks.append(mk.squeeze())

plot_multiple(ims, title="Imágenes Columbia")
plot_multiple(mks, title="Máscaras Columbia")
