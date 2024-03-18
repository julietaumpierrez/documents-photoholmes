import array
import glob
import os
from fileinput import filename

import numpy as np
from sympy import plot

from photoholmes.methods.factory import MethodFactory
from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.image import plot_multiple, read_image

IMAGE_PATH = "data/debug/splicebuster/Sp_D_CND_A_pla0005_pla0023_0281.jpg"
RERUN_METHOD = True

# Paths
DEBUG_ARRAYS_DIR = "data/debug/splicebuster/"
GROUND_TRUTHS = os.path.join(DEBUG_ARRAYS_DIR, "ground-truths/")
ATTEMPTS = os.path.join(DEBUG_ARRAYS_DIR, "attempts/")

# Method parameters
METHOD_NAME = "splicebuster"
CONFIG_PATH = "src/photoholmes/methods/splicebuster/config.yaml"


def compare_arrays(true_array, estimated_array, threshold, array_name=""):
    print("-------------------------------------------------------")
    print("\n\nAssesing file:", array_name)
    if true_array.shape != estimated_array.shape:
        print(
            f"Arrays {array_name} have different shapes: {true_array.shape} and {estimated_array.shape}"
        )
        print(true_array)
    else:
        if np.allclose(true_array, estimated_array):
            print(f"{array_name}: Arrays numéricamente iguales")
        else:
            print(
                f"{array_name}: Proporción de elementos con distancia >{threshold}:",
                (np.abs(true_array - estimated_array) > threshold).mean(),
            )
            print(
                f"{array_name}: Maxima diferencia relativa:",
                (np.abs(true_array - estimated_array)).max()
                / np.abs(true_array).mean(),
            )
            # print(
            #     "Verdadero:\n",
            #     true_array,
            #     "\nEstimado:\n",
            #     estimated_array,
            #     "\nDiferencia:\n",
            #     true_array - estimated_array,
            # )
            (
                print(f"Arrays cercanos con tolerancia {threshold}.")
                if np.allclose(true_array, estimated_array, atol=threshold)
                else print("Arrays distintos")
            )


def normalize(array):
    return ZeroOneRange()(array)["image"]


config = load_yaml(CONFIG_PATH)
im = read_image(IMAGE_PATH)
if RERUN_METHOD:
    # Run method over images
    method, preprocess = MethodFactory.load(METHOD_NAME, config)

    im_preprocessed = preprocess(image=im)
    out = method.benchmark(**im_preprocessed)
    heatmap = out["heatmap"]
    heatmap = heatmap.numpy()
else:
    heatmap = np.load(os.path.join(ATTEMPTS, "heatmap.npy"))
true_heatmap = np.load(os.path.join(GROUND_TRUTHS, "heatmap.npy"))

compare_arrays(normalize(true_heatmap), normalize(heatmap), 1e-2, "heatmap")
plot_multiple([im, heatmap, true_heatmap])
