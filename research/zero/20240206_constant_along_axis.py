# %%
import numpy as np

SHAPE = (16, 16)
image = np.empty(SHAPE)
line = [range(8)]
image[:8, :8] = np.array([range(8)] * 8)
image[:8, 8:] = np.array([range(8, 16)] * 8).T
image[8:, :8] = np.array([range(8)] * 8).T
image[8:, 8:] = np.array([range(8, 16)] * 8)
print(image)
luminance = image.copy()
# %%
x, y = 8, 0
block = luminance[x : x + 8, y : y + 8]
print(
    block,
    "\n\n",
    block[:1, :],
    "\n",
    block[:, :1],
)
const_along_x = np.all(block[:, :] == block[:1, :])
const_along_y = np.all(block[:, :] == block[:, :1])

print(const_along_x)
print(const_along_y)
