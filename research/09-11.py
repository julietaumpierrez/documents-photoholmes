# %%
# FIXME: No corre acá, solo adentro de src.
import os
import cv2 as cv
import matplotlib.pyplot as plt

from photoholmes.utils import image
from photoholmes.models import Naive

DATA_DIR = 'benchmarking/test_images/'

IMAGES_PATH = DATA_DIR + 'images/'
MASK_PATH = DATA_DIR + 'masks/'

# %%
images = [cv.imread(IMAGES_PATH+path) for path in os.listdir(IMAGES_PATH)]
image.plot_multiple_images(images=images, titles=os.listdir(IMAGES_PATH), ncols=2)

# %%
image_choice = 1
method = Naive()

name = f'Im_{image_choice}'
im = cv.imread(IMAGES_PATH+name+'.jpg')
mask = cv.imread(MASK_PATH+name+'.png')

method = Naive()
heatmap = method.predict(im)
predicted_mask = method.predict_mask(im)

#%%
fig, ax = plt.subplots(1,4)
ax[0].imshow(im)
ax[0].set_title('Imagen')
ax[0].set_axis_off()
ax[1].imshow(heatmap)
ax[1].set_title('Heatmap')
ax[1].set_axis_off()
ax[2].imshow(predicted_mask)
ax[2].set_title('Predicted Mask')
ax[2].set_axis_off()
ax[3].imshow(mask)
ax[3].set_title('GT Mask')
ax[3].set_axis_off()
plt.tight_layout()
plt.show()

# %%
