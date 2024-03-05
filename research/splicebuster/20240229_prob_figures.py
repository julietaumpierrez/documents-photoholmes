import os

import matplotlib.pyplot as plt
import numpy as np

FILENAME = "loss"

ATTEMPTS = "data/debug/splicebuster/attempts/"
GROUND_TRUTHS = "data/debug/splicebuster/ground-truths/"
PROB_SERIES_PATH = "data/debug/splicebuster/debug_series/"
ATTEMPTS_FOLDER = f"attempt/"
GROUND_TRUTHS_FOLDER = "ground-truth/"
FILENAME += ".npy"

gt_probs = np.load(os.path.join(PROB_SERIES_PATH, GROUND_TRUTHS_FOLDER, FILENAME))
print(gt_probs.shape)
gt_probs = gt_probs.squeeze()

at_probs = np.load(os.path.join(PROB_SERIES_PATH, ATTEMPTS_FOLDER, FILENAME))
print(at_probs.shape)


if FILENAME == "nlogl.npy":
    close = np.allclose(gt_probs[:-1, :, 0], at_probs[:-1], atol=1e0)
    print("Arrays close: ", close)
    # if not close:
    difference = gt_probs[..., 0] - at_probs

    im = np.exp(difference.copy() * 1000)
    plt.figure()
    plt.imshow(im)
    plt.colorbar()
    plt.legend()
    plt.show()
    # else:
    plt.figure()
    plt.plot(gt_probs[-1, :, 0] - at_probs[-1, :], label="Ground truth")
    plt.legend()
    plt.show()

if FILENAME == "pi.npy":
    print("Arrays close: ", np.allclose(gt_probs, at_probs, atol=1e-5))
    plt.figure()
    plt.plot(at_probs, label="Estimated")
    plt.plot(gt_probs, label="Ground truth")
    plt.legend()
    plt.show()

if FILENAME == "loss.npy":
    print("Arrays close: ", np.allclose(gt_probs, at_probs, atol=1e-5))
    plt.figure()
    plt.plot(at_probs, label="Estimated")
    plt.plot(gt_probs, label="Ground truth")
    plt.legend()
    plt.show()

if FILENAME == "mean.npy" or FILENAME == "covariance.npy":
    print(np.allclose(gt_probs, np.load(GROUND_TRUTHS + FILENAME), atol=1e-5))
    diffs = gt_probs - at_probs
    close = np.allclose(diffs, np.zeros_like(diffs), atol=1e-5)
    print("Arrays close: ", close)
    if not close:
        print(diffs)
        im = diffs.copy()
        plt.figure()
        plt.imshow(im)
        plt.colorbar()
        plt.legend()
        plt.show()
