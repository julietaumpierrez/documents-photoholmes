import os

import numpy as np

from photoholmes.methods.method_factory import MethodFactory
from photoholmes.utils.image import plot_multiple, read_image

REPO_DIR = "/home/dsense/extra/tesis/photoholmes/extra/zero/zero"
IMAGES = ["tampered1.png", "tampered1.jpg", "tampered1_99.jpg"]
IMAGE = IMAGES[0]


def compare_arrays(true_array, estimated_array, threshold, array_name=""):
    print(
        f"{array_name}: Cantidad de elementos con distancia >{threshold}:",
        (np.abs(true_array - estimated_array) > threshold).sum(),
    )
    print(
        f"{array_name}: Maxima diferencia:",
        (np.abs(true_array - estimated_array)).max(),
    )
    print(true_array, estimated_array, true_array - estimated_array)
    (
        print("Arrays cercanos")
        if np.allclose(true_array, estimated_array, atol=threshold)
        else print("Arrays distintos")
    )


method_name = "zero"
method, preprocess = MethodFactory.load(method_name)

im = read_image(os.path.join(REPO_DIR, IMAGE))
im_preprocessed = preprocess(image=im)
out = method.predict(**im_preprocessed)

mask = out["mask"]

true_mask = np.loadtxt("data/debug/true_forgery_mask.csv", delimiter=",")
compare_arrays(mask, true_mask, 1, "forgery_mask")
plot_multiple([mask, true_mask])
