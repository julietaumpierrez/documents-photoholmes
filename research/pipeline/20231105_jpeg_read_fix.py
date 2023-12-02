import os
from tempfile import NamedTemporaryFile

import cv2 as cv
import numpy as np
from PIL import Image

from photoholmes.utils.image import _DCT_from_jpeg, plot, plot_multiple, read_jpeg_data

if "research" in os.path.abspath("."):
    os.chdir("../../")

jpg_path = "benchmarking/test_images/images/Im_1.jpg"
tif_path = (
    "/home/dsense/extra/tesis/datos/columbia/4cam_splc/canong3_canonxt_sub_01.tif"
)

dct, qtables = read_jpeg_data(jpg_path)
print(dct)
print(qtables)

dct, qtables = read_jpeg_data(tif_path)
print(dct)
print(qtables)
