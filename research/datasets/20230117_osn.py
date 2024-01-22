# %%
import os

from pyparsing import C
from sympy import plot

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from photoholmes.datasets.casia1 import (
    Casia1CopyMoveDataset,
    Casia1CopyMoveOSNDataset,
    Casia1SplicingDataset,
    Casia1SplicingOSNDataset,
)
from photoholmes.datasets.columbia import ColumbiaDataset, ColumbiaOSNDataset
from photoholmes.datasets.coverage import CoverageDataset
from photoholmes.datasets.dso1 import DSO1Dataset, DSO1OSNDataset
from photoholmes.utils.image import plot_multiple

# %%
DSO1_PATH = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/"
dataset = ColumbiaOSNDataset(DSO1_PATH, tampered_only=True)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk, names = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes DSO1")
plot_multiple(mks, title="Máscaras DSO1")

# %%
# Check if masks have the same shape as images
for n in range(len(dataset)):
    x, mk, names = dataset[n]
    if x["image"].shape[1] != mk.shape[0] and x["image"].shape[2] != mk.shape[1]:
        print(names)
# %%
x, mk, names = dataset[0]
print(x["image"].shape)
print(mk.shape)
# %%
