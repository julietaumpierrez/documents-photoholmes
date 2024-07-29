# %%
import os
from pathlib import Path

import numpy as np
from PIL import Image

from photoholmes.methods.noisesniffer import Noisesniffer, noisesniffer_preprocessing
from photoholmes.utils.image import read_image, read_jpeg_data

method = Noisesniffer()
preprocesing = noisesniffer_preprocessing

out_folder = Path(
    f"/Users/julietaumpierrez/Desktop/images_for_eval_outputs/{method.__class__.__name__}"
)
images_folder = Path("/Users/julietaumpierrez/Desktop/new_eval_trace")

os.makedirs(out_folder, exist_ok=True)


image_list = os.listdir(images_folder)
print(len(image_list))
for image_file in image_list:
    print(image_file)
    image = read_image(str(images_folder / image_file))
    dct, qt = read_jpeg_data(str(images_folder / image_file))

    inp = preprocesing(image=image, dct_coefficients=dct, quantization_table=qt)

    output = method.predict(**inp)

    mask = Image.fromarray(output[0] * 255).convert("L")

    mask.save(out_folder / f"{image_file}.pdf")

# %%
