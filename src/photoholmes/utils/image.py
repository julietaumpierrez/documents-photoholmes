import imghdr
import logging
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

logger = logging.getLogger(__name__)


def read_image(path) -> torch.Tensor:
    return torch.from_numpy(
        cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB).transpose(2, 0, 1)
    )


def save_image(path, img: torch.Tensor | np.ndarray, *args):
    if isinstance(img, torch.Tensor):
        img_bgr = cv.cvtColor(tensor2numpy(img), cv.COLOR_RGB2BGR)
    else:
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(path, img_bgr, *args)


def tensor2numpy(image: torch.Tensor) -> np.ndarray:
    img = image.numpy()
    return img.transpose(1, 2, 0) if image.ndim > 2 else img


def plot(image: torch.Tensor | np.ndarray, title=None, save_path=None):
    """Function for easily plotting an image."""
    if isinstance(image, torch.Tensor):
        image = tensor2numpy(image)
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis(False)
    if save_path is not None:
        plt.savefig(save_path)
        print("Figure saved at:", save_path)
    plt.show()


def plot_multiple(
    images,
    titles=None,
    ncols=4,
    title: Optional[str] = None,
    save_path=None,
):
    """Function for easily plotting one or multiple images"""
    N = len(images)
    nrows = np.ceil(N / ncols).astype(int)
    if titles is None:
        titles = [None] * len(images)
    if nrows > 1:
        fig, ax = plt.subplots(nrows, ncols)
        for n, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = tensor2numpy(img)
            i = n // ncols
            j = n % ncols
            ax[i, j].imshow(img)
            ax[i, j].set_title(titles[n])
            ax[i, j].set_axis_off()
    else:
        fig, ax = plt.subplots(1, N)
        for n, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = tensor2numpy(img)
            ax[n].imshow(img)
            ax[n].set_title(titles[n])
            ax[n].set_axis_off()
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print("Figure saved at:", save_path)
    plt.show()


def read_mask(mask_path):
    """Returns mask as a boolean image, from a mask path"""
    mask = cv.imread(mask_path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    return mask > mask.max() / 2


def read_jpeg_data(
    image_path: str,
    num_dct_channels: Optional[int] = None,
    all_quant_tables: bool = False,
    suppress_not_jpeg_warning: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Reads image from path and returns DCT coefficient matrix for each channel and the
    quantization matrixes. If image is in jpeg format, it decodes the DCT stream and
    returns it. Otherwise, the image is saved into a temporary jpeg file and then the
    DCT stream is decoded.

    Parameters:
        image_path: Path to image
        n_channels: Number of channels to read. If 1, only Y channel is read.
        quant_tables:
    Returns:
        dct: DCT coefficient matrix for each channel
        qtables: Quantization matrix for each channel
    """
    if imghdr.what(image_path) == "jpeg":
        jpeg = jpegio.read(image_path)
    else:
        if not suppress_not_jpeg_warning:
            logger.warning(
                "Image is not in JPEG format. An approximation will be loaded by "
                "compressing the image in quality 100."
            )
        temp = NamedTemporaryFile(suffix=".jpg")
        img = read_image(image_path)
        save_image(temp.name, img, [cv.IMWRITE_JPEG_QUALITY, 100])
        jpeg = jpegio.read(temp.name)

    return torch.tensor(
        _DCT_from_jpeg(jpeg, num_channels=num_dct_channels)
    ), torch.tensor(np.array(_qtables_from_jpeg(jpeg, all=all_quant_tables)))


def _qtables_from_jpeg(
    jpeg: jpegio.DecompressedJpeg, all: bool = False
) -> List[NDArray]:
    if all:
        return [jpeg.quant_tables[i].copy() for i in range(len(jpeg.quant_tables))]
    else:
        return [jpeg.quant_tables[0].copy()]


def _DCT_from_jpeg(
    jpeg: jpegio.DecompressedJpeg, num_channels: Optional[int] = None
) -> np.ndarray:
    """
    :param im_path: JPEG image path
    :return: DCT_coef (Y,Cb,Cr)
    Code derived from https://github.com/mjkwon2021/CAT-Net.git.
    """
    if num_channels is None:
        num_channels = len(jpeg.coef_arrays)
    ci = jpeg.comp_info

    sampling_factors = np.array(
        [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
    )
    if num_channels == 3:
        if (sampling_factors[:, 0] == sampling_factors[0, 0]).all():
            sampling_factors[:, 0] = 2
        if (sampling_factors[:, 1] == sampling_factors[0, 1]).all():
            sampling_factors[:, 1] = 2
    else:
        sampling_factors[0, :] = 2

    dct_shape = jpeg.coef_arrays[0].shape
    DCT_coef = np.empty((num_channels, *dct_shape))

    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        block_coefs = (
            jpeg.coef_arrays[i].reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        )
        r_factor, c_factor = 2 // sampling_factors[i][0], 2 // sampling_factors[i][1]
        channel_coefficients = np.zeros((r * r_factor, c * c_factor))
        channel_coefficient_blocks = channel_coefficients.reshape(
            r // 8, r_factor * 8, c // 8, c_factor * 8
        ).transpose(0, 2, 1, 3)
        channel_coefficient_blocks[:, :, :, :] = np.tile(
            block_coefs, (r_factor, c_factor)
        )

        DCT_coef[i, :, :] = channel_coefficients[: dct_shape[0], : dct_shape[1]]

    return DCT_coef.astype(int)
