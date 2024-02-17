# type: ignore
# %%
from models.vit import FOCAL_ViT

# %%
model = FOCAL_ViT()

# %%
import torch
from models.hrnet import FOCAL_HRNet

model = FOCAL_HRNet()
hrnet_weights = torch.load(
    "weights/FOCAL_HRNet_weights.pth", map_location=torch.device("cpu")
)

hrnet_weights.keys()
# %%
model.load_state_dict(hrnet_weights["state_dict"])
# %%
model.state_dict
# %%
from main import FOCAL

# %%
focal = FOCAL(
    [
        ("ViT", "FOCAL_ViT_weights.pth"),
        ("HRNet", "FOCAL_HRNet_weights.pth"),
    ]
)
# %%
vit = focal.network_list[0]
# %%
focal_vit = vit.module
focal_vit
# %%
vit = focal_vit.net

# %%
torch.save(vit.state_dict(), "weights/ViT_weights.pth")

# %%
from torchvision.io import read_image

img = read_image("img00.png") / 255
img
# %%
from torchvision.transforms.functional import resize

img = resize(img, (1024, 1024))
# %%
out = focal.process(img[None, :], 0)
# %%
import matplotlib.pyplot as plt

plt.imshow(out[0][0])

# %%
from focal.models.vit import ImageEncoderViT

vit = ImageEncoderViT()
weights = torch.load("weights/ViT_weights.pth", map_location=torch.device("cpu"))
vit.load_state_dict(weights)

# %%
hrnet = focal.network_list[1].module.net
hrnet

# %%
torch.save(hrnet.state_dict(), "weights/HRNet_weights.pth")

# %%
from focal.models.hrnet import HRNet

hrnet = HRNet()
weights = torch.load("weights/HRNet_weights.pth", map_location=torch.device("cpu"))
hrnet.load_state_dict(weights)

# %%
from focal.method import Focal

focal = Focal(
    ["ViT", "HRNet"], ["weights/ViT_weights.pth", "weights/HRNet_weights.pth"]
)
# %%
img = read_image("hybrid_select_exo.png") / 255
mask = read_image("mask_exo.png") / 255
# img = resize(img, (1024, 1024))

# %%
out = focal.predict(img)

# %%
plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.imshow(out)
plt.subplot(1, 2, 2)
plt.imshow(mask[0])

# %%
