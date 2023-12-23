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

from photoholmes.datasets.trace import TraceHybridEndoDataset, TraceHybridExoDataset
from photoholmes.utils.image import plot_multiple

# %%
TRACE_PATH = "/Users/julietaumpierrez/Desktop/Datasets/trace/images"
dataset = TraceHybridExoDataset(TRACE_PATH)
print(len(dataset))
ims = []
mks = []
idxs = list(range(0, 6)) + list(range(len(dataset) - 6, len(dataset)))
for n in idxs:
    x, mk = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes Trace noise exo")
plot_multiple(mks, title="Máscaras Trace noise exo")

# %%
