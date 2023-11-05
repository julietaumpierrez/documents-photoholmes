import os
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple, Union

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from PIL.Image import open

IMG_FOLDER_PATH = "test_images/images/"


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
    image_path: str, num_dct_channels: Optional[int] = None
) -> Tuple[NDArray, List[NDArray]]:
    """Reads image from path and returns DCT coefficient matrix for each channel and the
    quantization matrixes. If image is in jpeg format, it decodes the DCT stream and
    returns it. Otherwise, the image is saved into a temporary jpeg file and then the
    DCT stream is decoded.

    Parameters:
        image_path: Path to image
        n_channels: Number of channels to read. If 1, only Y channel is read.
    Returns:
        dct: DCT coefficient matrix for each channel
        qtables: Quantization matrix for each channel
    """
    extension = (image_path[-4:]).lower()
    if extension == ".jpg" or extension == ".jpeg":
        jpeg = jpegio.read(image_path)
        return _DCT_from_jpeg(jpeg), _qtables_from_jpeg(jpeg)
    else:
        temp = NamedTemporaryFile(suffix=".jpg")
        img = read_image(image_path)
        save_image(temp.name, img, [cv.IMWRITE_JPEG_QUALITY, 100])
        jpeg = jpegio.read(temp.name)
        return _DCT_from_jpeg(jpeg), _qtables_from_jpeg(jpeg)


def _qtables_from_jpeg(
    jpeg: jpegio.DecompressedJpeg, num_channels: Optional[int] = None
) -> List[NDArray]:
    if num_channels is None:
        num_channels = len(jpeg.quant_tables)
    return [jpeg.quant_tables[i].copy() for i in range(num_channels)]


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
        sampling_factors[:, :] = 2

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


def non_overlapping_blocks(
    image: np.ndarray, block_shape: tuple = (8, 8)
) -> np.ndarray:
    """Divide an image into non-overlapping blocks of shape "block_shape".
    TODO: Add border cases when image shape is not divisible by block shape."""
    h, w, nc = image.shape
    bh, bw = block_shape
    strides = (w, 1, w, 1)
    blocked_image_shape = (h // bh, w // bw, bh, bw)
    blocks = np.empty((*blocked_image_shape, 3))
    print(blocks.shape)
    for channel in range(nc):
        blocks[:, :, :, :, channel] = np.lib.stride_tricks.as_strided(
            image[:, :, channel], shape=blocked_image_shape, strides=strides
        )
    return blocks


@dataclass
class ImFile:
    name: str
    img: np.ndarray
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_path(cls, image_path: str, mask_path: Optional[str] = None):
        """Initializes image from a given image_path, and optionally a mask path containing forgery ground truth."""
        name = image_path.split("/")[-1]
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = mask = cv.imread(mask_path) if mask_path is not None else None
        return cls(name, img, mask)

    @property
    def format(self):
        return self.name.split(".")[-1]

    def show(self, ax=None, save_path=None):
        """Displays the content of the image"""
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            self._plot_img(ax)
            if save_path is not None:
                plt.savefig(save_path)
            plt.show()
        else:
            self._plot_img(ax)

    def _plot_img(self, ax):
        """Plots an image with its title on an axis"""
        ax.imshow(self.img)
        ax.set_title(self.name)


class Data:
    def __init__(self, imfiles: list[ImFile], name: Optional[str] = None) -> None:
        self.imfiles = imfiles
        self.name = name

    @property
    def images(self) -> list[np.ndarray]:
        return [imfile.img for imfile in self.imfiles]

    @property
    def masks(self) -> list[Union[np.ndarray, None]]:
        return [imfile.mask for imfile in self.imfiles]

    @property
    def names(self) -> list[str]:
        return [imfile.name for imfile in self.imfiles]

    @property
    def format(self) -> str:
        """Returns image format, assuming all the images in 'path' share the format."""
        return self.imfiles[0].format

    @property
    def size(self) -> int:
        """Dataset size"""
        return len(self.imfiles)

    @classmethod
    def from_path(cls, img_folder_path: str, mask_folder_path: Optional[str] = None):
        """Initializes image Database from a folder path, and optionally masks from mask folder path"""
        name = cls._folder_name(img_folder_path)
        img_names = os.listdir(img_folder_path)
        if mask_folder_path is not None:
            return cls(
                [
                    ImFile.from_path(
                        img_folder_path + name, mask_folder_path + cls._mask_name(name)
                    )
                    for name in img_names
                ],
                name=name,
            )
        else:
            return cls(
                [ImFile.from_path(img_folder_path + name) for name in img_names],
                name=name,
            )

    @staticmethod
    def _mask_name(name: str) -> str:
        """Returns associated mask name from image name"""
        # TODO: Definir estándar de cómo se nombran una y la otra

        return "".join(name.split(".")[:-1]) + ".png"

    @staticmethod
    def _folder_name(path: str) -> str:
        return path.split("/")[-2]

    def show(self, ncols=4, title: Optional[str] = None, save_path=None):
        """Displays the dataset"""
        plot_multiple(
            self.images, self.names, ncols=ncols, title=title, save_path=save_path
        )


if __name__ == "__main__":
    test_base = Data.from_path(IMG_FOLDER_PATH)
    test_base.show(title="Prueba")
