```# %%
import os

os.chdir("..")
# %%
from photoholmes.methods.splicebuster import Splicebuster, splicebuster_preprocessing
from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing
from photoholmes.utils.image import read_image

import matplotlib.pyplot as plt
from PIL import Image

image = read_image("/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/4cam_splc/canong3_canonxt_sub_01.tif")
img = Image.fromarray(image.permute(1, 2, 0).numpy())
img.save("out/img00.pdf")
img
# %%
inp = splicebuster_preprocessing(image=image)

psccnet = PSCCNet(weights = {
        "FENet": "path_to_HRNet_weights",
        "SegNet": "path_to_NLCDetection_weights",
        "ClsNet": "path_to_DetectionHead_weights",
    })
output = psccnet.predict(**inp)
# %%
plt.imshow(output, cmap="gray")
plt.axis("off")
plt.savefig("out/mask.pdf")
# %%
mask = Image.fromarray(output * 255).convert("L")
mask.save("out/mask.pdf")
mask

# %%
mask_orig = read_image("../implementations/splicebuster/data/ref00.png") / 255

mask_orig = 1 - mask_orig
# %%
mask_orig = Image.fromarray(mask_orig.permute(1, 2, 0).numpy()[:, :, 0] * 255).convert("L")
mask_orig

# %%
mask_orig.min()

# %%
mask_orig.max()

# %%```