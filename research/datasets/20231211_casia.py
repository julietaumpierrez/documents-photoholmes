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

from photoholmes.datasets.casia1 import Casia1CopyMoveDataset, Casia1SplicingDataset
from photoholmes.datasets.coverage import CoverageDataset
from photoholmes.datasets.dso1 import DSO1Dataset
from photoholmes.utils.image import plot_multiple

# %%
CASIA_PATH = "/Users/julietaumpierrez/Desktop/Datasets/CASIA 1.0 dataset/"
dataset = Casia1CopyMoveDataset(CASIA_PATH)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes Casia1")
plot_multiple(mks, title="Máscaras Casia1")

# %%
CASIA_PATH = "/Users/julietaumpierrez/Desktop/Datasets/CASIA 1.0 dataset/"
dataset = Casia1SplicingDataset(CASIA_PATH)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes Casia1")
plot_multiple(mks, title="Máscaras Casia1")
# %%
DSO1_PATH = "/Users/julietaumpierrez/Desktop/Datasets/tifs-database/"
dataset = DSO1Dataset(DSO1_PATH)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes DSO1")
plot_multiple(mks, title="Máscaras DSO1")


# %%
COVERAGE_PATH = "/Users/julietaumpierrez/Desktop/Datasets/COVERAGE/"
dataset = CoverageDataset(COVERAGE_PATH)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes DSO1")
plot_multiple(mks, title="Máscaras DSO1")

# %%
