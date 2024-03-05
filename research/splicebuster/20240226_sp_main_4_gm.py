import array
import glob
import os
from fileinput import filename

import numpy as np
from sympy import plot

from photoholmes.methods.method_factory import MethodFactory
from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.image import plot_multiple, read_image

IMAGE_PATH = "data/debug/img03.png"
RERUN_METHOD = True

# Debug array paths
ARRAYS_TO_COMPARE = [
    "mean.npy",
    # "L.npy",
    "covariance.npy",
    # "Xc.npy",
    "heatmap.npy",
    # "mahalanobis.npy",
]  # ["pca_features.npy", "heatmap.npy"]
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


class LoadAndCompare:
    def __init__(self, gts_dir, attempts_dir, filenames_list) -> None:
        self.gts_dir = gts_dir
        self.attempts_dir = attempts_dir
        self.filenames_list = filenames_list

    def load(self):
        gt_arrays = [
            np.load(os.path.join(self.gts_dir, filename))
            for filename in self.filenames_list
        ]
        attempts_arrays = [
            np.load(os.path.join(self.attempts_dir, filename))
            for filename in self.filenames_list
        ]
        return gt_arrays, attempts_arrays

    def run(self, threshold=1e-12):
        gt_arrays, attempts_arrays = self.load()
        for gt_array, attempts_array, filename in zip(
            gt_arrays, attempts_arrays, self.filenames_list
        ):
            compare_arrays(gt_array, attempts_array, threshold, filename)


config = load_yaml(CONFIG_PATH)
im = read_image(IMAGE_PATH)
if RERUN_METHOD:
    # Run method over images
    method, preprocess = MethodFactory.load(METHOD_NAME, config)

    im_preprocessed = preprocess(image=im)
    out = method.predict(**im_preprocessed)
    heatmap = out["heatmap"]

# Load and compare
LoadAndCompare(GROUND_TRUTHS, ATTEMPTS, ARRAYS_TO_COMPARE).run(threshold=1e-3)

heatmap = np.load(os.path.join(ATTEMPTS, "heatmap.npy"))
true_heatmap = np.load(os.path.join(GROUND_TRUTHS, "heatmap.npy"))

# compare_arrays(normalize(true_heatmap), normalize(heatmap), 1e-2, "heatmap")
# pad_size = config["stride"]
# true_heatmap = np.pad(
#     true_heatmap,
#     ((pad_size, pad_size), (pad_size, pad_size)),
#     "constant",
#     constant_values=0,
# )
plot_multiple([im, heatmap, true_heatmap])
