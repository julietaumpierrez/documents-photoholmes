# %%
import os

from photoholmes.methods.DQ import method

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import torch
from torch.utils.data import DataLoader, Dataset

from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.utils.image import plot_multiple, read_image

# %%
CONFIG_PATH = "src/photoholmes/methods/psccnet/config.yaml"
METHOD_NAME = "psccnet"
method, pre = MethodFactory.load(METHOD_NAME, config=CONFIG_PATH)

# %%
PSCC_SAMPLE_PATH = "/home/dsense/extra/tesis/photoholmes/extra/PSCC-Net/sample"


class SampleDataset(Dataset):
    def __init__(self, sample_dir, transform) -> None:
        super().__init__()
        self.sample_dir = sample_dir
        self.transform = transform
        self.image_paths = os.listdir(sample_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        im_name = self.image_paths[index]
        image = read_image(os.path.join(self.sample_dir, im_name))
        x = {"image": image}
        if self.transform:
            x = self.transform(**x)
        return x


dataset = SampleDataset(PSCC_SAMPLE_PATH, transform=pre)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,  # must be 1 to handle arbitrary input sizes
    shuffle=False,  # must be False to get accurate filename
    num_workers=1,
    pin_memory=False,
)

ims = []
mks_pred = []
scores = []
# names = []
for x in list(loader):
    y = method.predict(**x)
    im = x["image"]
    # name = x["name"]
    # names.append(name)
    ims.append(im.squeeze())
    mks_pred.append(y["heatmap"].squeeze())
    scores.append(y["score"])

print(scores)
ims_to_plot = ims + mks_pred
titles = [None] * len(ims) + [f"Score: {s.item():.2f}" for s in scores]
plot_multiple(ims_to_plot, titles=titles, title="Imágenes Columbia")

# %%
# Columbia Test
# %%
COLUMBIA_PATH = "/home/dsense/extra/tesis/datos/columbia"
DEVICE = "cpu"

dataset = ColumbiaDataset(COLUMBIA_PATH, transform=pre)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,  # must be 1 to handle arbitrary input sizes
    shuffle=False,  # must be False to get accurate filename
    num_workers=1,
    pin_memory=False,
)

ims = []
mks = []
mks_pred = []
ys = []
for x, mk in list(loader)[:3]:
    y = method.predict(**x)
    im = x["image"]
    mk_pred = y["heatmap"]
    ys.append(y)
    ims.append(im.squeeze())
    mks_pred.append(mk_pred.squeeze())
    mks.append(mk.squeeze())

plot_multiple(ims + mks_pred + mks, ncols=3, title="Imágenes Columbia")
