# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # type: ignore
from skimage.util import view_as_windows

# %%
image = Image.open(
    "/Users/julietaumpierrez/Desktop/NoiseSniffer/test.png"
)  # .convert("L")
np_image = np.array(image)
plt.imshow(np_image, cmap="gray")
# %%
w = 3
ch = 1
img_blocks = view_as_windows(np_image[:, :, ch], w).reshape(-1, w, w)
# %%
print(img_blocks.shape)
# %%
print(img_blocks[0])
# %%
plt.imshow(img_blocks[0])
# %%
plt.imshow(np_image[:3, :3, 1])
# %%
