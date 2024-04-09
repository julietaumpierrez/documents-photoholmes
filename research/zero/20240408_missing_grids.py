# %%
import os

if "research" in os.getcwd():
    os.chdir("../..")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from photoholmes.utils.image import read_image

# %%
img = read_image("data/COVERAGE/image/28t.tif")

# %%
import time
from tempfile import NamedTemporaryFile

import cv2

t0 = time.time()
f = NamedTemporaryFile(suffix=".jpg")
cv2.imwrite(f.name, img.numpy().transpose(1, 2, 0), [cv2.IMWRITE_JPEG_QUALITY, 99])
img_99 = cv2.imread(f.name)
f.close()
tf = time.time()
print("Elapsed time:", tf - t0)

# %%
from io import BytesIO

import numpy as np
from PIL import Image

t0 = time.time()
img_pil = Image.fromarray(img.numpy().transpose(1, 2, 0))
f = BytesIO()
img_pil.save(f, "JPEG", quality=99)
f.seek(0)
img_99_pil = Image.open(f)
img_99 = np.array(img_99_pil)
f.close()
t1 = time.time()
print("Elapsed time:", t1 - t0)

# %%
t0 = time.time()
img = cv2.imread("data/COVERAGE/image/28t.tif")
t1 = time.time()
print("CV2 time:", t1 - t0)
t0 = time.time()
img_pil = Image.open("data/COVERAGE/image/28t.tif")
img = np.array(img_pil)
t1 = time.time()
print("PIL time:", t1 - t0)

from io import BytesIO

# %%
from typing import Any, Dict

from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from photoholmes.preprocessing import BasePreprocessing


class AddImage99(BasePreprocessing):

    def __call__(self, image: Tensor | NDArray, **kwargs) -> Dict[str, Any]:

        if isinstance(image, Tensor):
            image = image.permute(1, 2, 0).numpy()

        img_pil = Image.fromarray(image)

        f = BytesIO()
        img_pil.save(f, "JPEG", quality=99)
        f.seek(0)
        image_99 = Image.open(f)
        image_99 = np.array(image_99)
        f.close()

        return {"image": image, "image_99": image_99, **kwargs}


# %%
import matplotlib.pyplot as plt

from photoholmes.methods.zero import zero_preprocessing

inp = zero_preprocessing(image=img)
inp.keys()

plt.imshow(inp["image_99"])
plt.show()
plt.imshow(inp["image"])
plt.show()
# %%
np.allclose(inp["image"], img)

# %%
from photoholmes.methods.zero import Zero

zero = Zero()

# %%
img = read_image("data/COVERAGE/image/21.tif")
# img = read_image("../implementations/zero/roma.png")
inp = zero_preprocessing(image=img)
pred, missing_grids = zero.predict(**inp)

# %%
plt.imshow(pred)
# %%
plt.imshow(missing_grids)

# %%
missing_grids
# %%
img = Image.open("data/COVERAGE/image/21.tif")
img.save("data/21_pil.jpeg", "JPEG", quality=99)
img
# %%
from photoholmes.utils.image import save_image

img = read_image("data/COVERAGE/image/21.tif")
plt.imshow(img.permute(1, 2, 0))
save_image("data/21_cv.jpeg", img, [cv2.IMWRITE_JPEG_QUALITY, 99])

# %%
img_pil = Image.open("data/21.jpeg")
plt.imshow(img_pil)
plt.show()
img_cv = Image.open("data/21_cv.jpeg")
plt.imshow(img_cv)
plt.show()
assert np.allclose(np.array(img_pil), np.array(img_cv))

# %%
plt.imshow(np.abs(np.array(img_pil) - np.array(img_cv)))
# %%
img_magick = Image.open("data/21_magick.jpg")
plt.imshow(img_magick)
# %%
plt.imshow(np.abs(np.array(img_pil) - np.array(img_magick)))
plt.show()
plt.imshow(np.abs(np.array(img_cv) - np.array(img_magick)))
# %%
img = Image.open("../implementations/zero/tampered1.png")
img.save("../implementations/zero/tampered1_pil.jpeg", "JPEG", quality=99)

# %%
img = read_image("data/COVERAGE/image/21.tif")

preprocess_99 = AddImage99()

inp = preprocess_99(image=img)
assert (inp["image_99"] == np.array(Image.open("data/21_cv.jpeg"))).all()

# %%
img = read_image("data/COVERAGE/image/21.tif")
inp = zero_preprocessing(image=img)

pred, missing_grids = zero.predict(**inp)
# %%
plt.imshow(missing_grids)

# %%
from photoholmes.preprocessing import (
    PreProcessingPipeline,
    RGBtoGray,
    RoundToUInt,
    ToNumpy,
)

img = read_image("data/COVERAGE/image/21.tif")
img99 = read_image("data/21_pil.jpeg")
preprocess = PreProcessingPipeline(
    transforms=[
        RGBtoGray(extra_image_keys=["image_99"]),
        RoundToUInt(apply_on=["image", "image_99"]),
        ToNumpy(image_keys=["image", "image_99"]),
    ],
    inputs=["image", "image_99"],
    outputs_keys=["image", "image_99"],
)

inp = preprocess(image=img, image_99=img99)

# %%
plt.imshow(inp["image_99"])
# %%
pred, missing_grids = zero.predict(**inp)
# %%
plt.imshow(missing_grids, cmap="gray")

# %%
orig_mask = read_image("../implementations/zero/test2/mask_m.png")
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(orig_mask.permute(1, 2, 0), cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(missing_grids, cmap="gray")

# %%
lum_99_orig = read_image("../implementations/zero/test2/luminance99.png")
lum_99_orig = lum_99_orig[0].numpy()
# %%
plt.imshow(lum_99_orig, cmap="gray")
# %%
(inp["image_99"][..., 0] == lum_99_orig).all()

# %%
m, mm, l, l99 = zero.predict(**inp)
# %%
