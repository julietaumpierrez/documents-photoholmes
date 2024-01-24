# %%
import os
from re import T

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

from photoholmes.datasets.trace.trace_hybrid import (
    TraceHybridEndoDataset,
    TraceHybridExoDataset,
)
from photoholmes.utils.image import plot_multiple

# %%
# TRACE_PATH = "/Users/julietaumpierrez/Desktop/Datasets/trace/images"
TRACE_PATH = "data/Benchmark/MiniTrace"
dataset = TraceHybridEndoDataset(TRACE_PATH)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk, _ = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes Trace noise exo")
plot_multiple(mks, title="Máscaras Trace noise exo")

from photoholmes.datasets.trace import TraceCFAGridExoDataset as Dataset

# %%
from photoholmes.utils.image import plot_multiple

TRACE_PATH = "data/Benchmark/MiniTrace"
dataset = Dataset(TRACE_PATH)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk, _ = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title=f"{dataset.__class__.__name__} images")
plot_multiple(mks, title=f"{dataset.__class__.__name__} masks")
# %%
