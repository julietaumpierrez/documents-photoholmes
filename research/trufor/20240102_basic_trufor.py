# %%
import os

if "research" in os.getcwd():
    os.chdir("..")
    os.chdir("..")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
# %%
import torch

from photoholmes.methods.trufor.method import TruFor

method = TruFor()

ckpt = torch.load("weights/trufor.pth.tar", map_location="cpu")
method.load_state_dict(ckpt["state_dict"])
method.eval()
# %%
import matplotlib.pyplot as plt

from photoholmes.utils.image import read_image

img = read_image("data/img00.png") / 255
plt.imshow(img.permute(1, 2, 0))

# %%
with torch.no_grad():
    out, conf, det, npp = method.forward(img.unsqueeze(0))

# %%
plt.imshow(out[0, 0, :, :].cpu().numpy())
plt.show()
out0 = out[0, 0, :, :].cpu()
print(out0.min(), out0.max())
# %%
plt.imshow(out0 > 0)
# %%
plt.imshow(out[0, 1, :, :].cpu().numpy())
out1 = out[0, 1, :, :].cpu()
print(out1.min(), out1.max())
plt.show()
# %%
plt.imshow(out1 > 0)
# %%
sum_0 = torch.sum(out[:, 0, :, :])
sum_1 = torch.sum(out[:, 1, :, :])
print(sum_0, sum_1)
sum_maps = torch.sum(out, dim=[-1, -2])
print(sum_maps)
# %%
plt.imshow(out[0, torch.argmin(sum_maps[0, :]), :, :].cpu().numpy())
# %%
out = method.predict(img)
plt.imshow(out["heatmap"][0])
# %%
plt.imshow(out["confidence"][0, 0])

# %%
print(out["detection"])

# %%
auth_image = read_image("data/IMD2020/1a1ogs/1a1ogs_orig.jpg") / 255
plt.imshow(auth_image.permute(1, 2, 0))

# %%
out_auth = method.predict(auth_image)

# %%
plt.imshow(out_auth["heatmap"][0])
# %%
plt.imshow(out_auth["confidence"][0][0])
# %%
print(out_auth["detection"])

# %%
method = TruFor()

method.load_weights("weights/trufor.pth.tar")
# %%
out = method.predict(img)
plt.imshow(out["heatmap"][0])
