# %%
import cv2 as cv

from photoholmes.models import Naive
import matplotlib.pyplot as plt


IMAGE_DIR = 'benchmarking/test_images/'

# %%
choice = 1

name = f'Im_{choice}'
im = cv.imread(IMAGE_DIR+'images/'+name+'.jpg')
mask = cv.imread(IMAGE_DIR+'masks/'+name+'.png')


# %%
method = Naive()
prediction = method.predict_img(im)

fig, ax = plt.subplots(1,3)
ax[0].imshow(im)
ax[0].set_title('Imagen')
ax[0].set_axis_off()
ax[1].imshow(prediction)
ax[1].set_title('Prediction')
ax[1].set_axis_off()
ax[2].imshow(mask)
ax[2].set_title('Mask')
ax[2].set_axis_off()
plt.tight_layout()
plt.show()

# %%
