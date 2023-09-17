import numpy as np


def normalize_image(image):
    """ """
    return np.round(255 * (image - image.min()) / (image.max() - image.min())).astype(
        int
    )
