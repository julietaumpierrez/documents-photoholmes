from typing import Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


def upscale_mask(
    coords: Tuple[NDArray, NDArray],
    mask: NDArray,
    target_size: Tuple[int, int],
    method: Literal[
        "linear", "nearest", "slinear", "cubic", "quintic", "pchip"
    ] = "linear",
    fill_value: Union[int, float] = 0,
) -> NDArray:
    """Upscale a mask to a target size.
    Params:
        coords: coordinates of the mask values
        mask: mask to upscale
        target_size: target size
        method: interpolation method
        fill_value: value to fill outside the mask
    Returns:
        upscaled mask
    """
    X, Y = target_size
    interpolator = RegularGridInterpolator(
        coords, mask, method=method, bounds_error=False, fill_value=fill_value
    )
    target_coords = np.asarray(
        np.meshgrid(
            np.arange(0, X),
            np.arange(0, Y),
        )
    )
    return interpolator(target_coords.reshape(2, -1).T).reshape(Y, X).T
