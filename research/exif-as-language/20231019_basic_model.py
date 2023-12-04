# %%
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import DistilBertConfig, DistilBertModel

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")


# %%
checkpoint = torch.load("weights/exif/wrapper_75_new.pth", map_location="cpu")

# %%
checkpoint["model"].keys()


# %%
class Exif(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = resnet50()
        self.visual.fc = nn.Linear(2048, 768)
        bert_config = DistilBertConfig()
        self.transformer = DistilBertModel(bert_config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


model = Exif()
model.eval()
# %%
model.load_state_dict({"state_dict": checkpoint["model"]})

# %%
checkpoint["model"].keys()
# %% convert weights
new_weights = {}
for k, v in checkpoint["model"].items():
    new_weights[k.replace("model.", "")] = v

# %%
model.load_state_dict(new_weights)
# %%
new_weights.keys()
# %%
new_weights.pop("positional_embedding")
new_weights.pop("ln_final.weight")
new_weights.pop("ln_final.bias")
new_weights.pop("token_embedding.weight")
new_weights.pop("text_projection")
new_weights.pop("sink_temp")
# %%
model.load_state_dict(new_weights)

# %%
import torchvision

img = torchvision.io.read_image("data/img00.png")

# %%
with torch.no_grad():
    mask = model.visual(img.unsqueeze(0).float())

# %%
torch.save(model.state_dict(), "weights/exif/pruned_weights.pth")
# %%
a = torch.load("weights/exif/pruned_weights.pth")
# %%
model.state_dict()

# %%
