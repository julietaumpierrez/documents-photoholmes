# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.datasets.autosplice import (
    Autosplice75Dataset,
    Autosplice90Dataset,
    Autosplice100Dataset,
)
from photoholmes.methods.dq import DQ, dq_preprocessing

# %%
dataset = Autosplice75Dataset(
    "/Users/sote/Desktop/data/Datasets/AutoSplice/",
    preprocessing_pipeline=dq_preprocessing,
    item_data=["image"],
)

for image_data, mask, image_name in dataset:
    print(image_data, mask, image_name)
    break

# %%

dataset = Autosplice90Dataset(
    "/Users/sote/Desktop/data/Datasets/AutoSplice/",
    preprocessing_pipeline=dq_preprocessing,
    item_data=["image"],
)

for image_data, mask, image_name in dataset:
    print(image_data, mask, image_name)
    break

# %%

dataset = Autosplice100Dataset(
    "/Users/sote/Desktop/data/Datasets/AutoSplice/",
    preprocessing_pipeline=dq_preprocessing,
    item_data=["image"],
)

for image_data, mask, image_name in dataset:
    print(image_data, mask, image_name)
    break


# %%
from photoholmes.datasets.casia1 import Casia1CopyMoveDataset, Casia1SplicingDataset
from photoholmes.datasets.osn import Casia1CopyMoveOSNDataset, Casia1SplicingOSNDataset

# %%
dataset = Casia1SplicingDataset(
    "/Users/sote/Desktop/data/Datasets/CASIA_1/",
    preprocessing_pipeline=dq_preprocessing,
    item_data=["image"],
)
# %%
for image_data, mask, image_name in dataset:
    print(image_data, mask, image_name)
    break
# %%
dataset = Casia1SplicingOSNDataset(
    "/Users/sote/Desktop/data/Datasets/osn/",
    preprocessing_pipeline=dq_preprocessing,
    item_data=["image"],
)
# %%
for image_data, mask, image_name in dataset:
    print(image_data, mask, image_name)
    break
# %%
dataset = Casia1CopyMoveDataset(
    "/Users/sote/Desktop/data/Datasets/CASIA_1/",
    preprocessing_pipeline=dq_preprocessing,
    item_data=["image"],
)

# %%
for image_data, mask, image_name in dataset:
    print(image_data, mask, image_name)
    break
# %%
dataset = Casia1CopyMoveOSNDataset(
    "/Users/sote/Desktop/data/Datasets/osn/",
    preprocessing_pipeline=dq_preprocessing,
    item_data=["image"],
)
# %%
for image_data, mask, image_name in dataset:
    print(image_data, mask, image_name)
    break

# %%
