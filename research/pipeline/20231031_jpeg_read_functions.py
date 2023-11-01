import os
from tempfile import NamedTemporaryFile

import cv2 as cv
import numpy as np
from PIL import Image

from photoholmes.utils.image import _DCT_from_jpeg, plot, plot_multiple

if "research" in os.path.abspath("."):
    os.chdir("../../")

# PATH = "benchmarking/test_images/images/Im_1.jpg"
PATH = "/home/dsense/extra/tesis/datos/columbia/4cam_splc/canong3_canonxt_sub_01.tif"

im_cv = cv.cvtColor(cv.imread(PATH), cv.COLOR_BGR2RGB)
im_pil = Image.open(PATH)
plot_multiple([im_cv, im_pil])

temp_cv = NamedTemporaryFile(suffix=".jpg")
cv.imwrite(
    temp_cv.name, cv.cvtColor(im_cv, cv.COLOR_RGB2BGR), [cv.IMWRITE_JPEG_QUALITY, 100]
)

temp_pil = NamedTemporaryFile(suffix=".jpg")
im_pil.save(temp_pil.name, quality=100)

dct_pil = _DCT_from_jpeg(temp_pil.name)
dct_cv = _DCT_from_jpeg(temp_cv.name)

assert (dct_pil == dct_cv).all()
