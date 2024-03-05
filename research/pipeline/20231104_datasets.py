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

from photoholmes.datasets.realistic_tampering_webp import RealisticTamperingWebPDataset
from photoholmes.methods.DQ import DQ
from photoholmes.utils.image import plot_multiple

# Elegir Dataset
DATASET_NAME = "Realistic Tampering WebP"
DATA_PATH = "/home/dsense/extra/tesis/datos/"
FOLDER_NAME = "_".join((DATASET_NAME.lower()).split(" "))  # Replace if necessary
DATASET_PATH = DATA_PATH + FOLDER_NAME
print(DATASET_PATH)

dataset_class = RealisticTamperingWebPDataset

# %%
SEED = 42
dataset = dataset_class(DATASET_PATH)
ims = []
mks = []
rnd = np.random.default_rng(SEED)
idxs = rnd.choice(range(len(dataset)), 12).astype(int)
labels = []
for n in idxs:
    x, mk, name = dataset[n]
    ims.append(x["image"])
    mks.append(mk)
    labels.append(name)

plot_multiple(ims, titles=labels, title="Imágenes " + DATASET_NAME)
plot_multiple(mks, titles=labels, title="Máscaras " + DATASET_NAME)

# %%
