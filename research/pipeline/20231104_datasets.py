# %%
import os

from sympy import plot

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader

from photoholmes.data.input.reaslistic_tampering import RealisticTamperingDataset
from photoholmes.models.splicebuster import Splicebuster
from photoholmes.utils.image import plot_multiple

DATASET_NAME = "Realistic Tampering"
DATA_PATH = "/home/dsense/extra/tesis/datos/"
FOLDER_NAME = "_".join((DATASET_NAME.lower()).split(" "))  # Replace if necessary
DATASET_PATH = DATA_PATH + FOLDER_NAME

# %%
SEED = 42
dataset = RealisticTamperingDataset(DATASET_PATH)

ims = []
mks = []
rnd = np.random.default_rng(SEED)
idxs = rnd.choice(range(len(dataset)), 12).astype(int)

for n in idxs:
    x, mk = dataset[n]
    ims.append(x["image"])
    mks.append(mk)

plot_multiple(ims, title="Imágenes " + DATASET_NAME)
plot_multiple(mks, title="Máscaras " + DATASET_NAME)
