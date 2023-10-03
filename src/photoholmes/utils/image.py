import os
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Optional, Union

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import open

IMG_FOLDER_PATH = "test_images/images/"


def plot_multiple_images(
    images, titles=None, ncols=4, title: Optional[str] = None, save_path=None
):
    """D"""
    N = len(images)
    nrows = np.ceil(N / ncols).astype(int)
    fig, ax = plt.subplots(nrows, ncols)
    if titles is None:
        titles = [None] * len(images)
    if nrows > 1:
        for n, img in enumerate(images):
            i = n // ncols
            j = n % ncols
            ax[i, j].imshow(img)
            ax[i, j].set_title(titles[n])
    else:
        for n, img in enumerate(images):
            ax[n].imshow(img)
            ax[n].set_title(titles[n])
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


def read_DCT(image_path: str) -> np.ndarray:
    """Reads image from path and returns DCT coefficient matrix for each channel.
    If image is in jpeg format, it decodes the DCT stream and returns it.
    Otherwise, the image is saved into a temporary jpeg file and then the DCT stream is decoded.
    """
    extension = (image_path[-4:]).lower()
    if extension == ".jpg" or extension == ".jpeg":
        return _DCT_from_jpeg(image_path)
    else:
        temp = NamedTemporaryFile(suffix=".jpg")
        open(image_path).convert("RGB").save(temp.name, quality=100, subsampling=0)
        return _DCT_from_jpeg(temp.name)


def _DCT_from_jpeg(image_path: str) -> np.ndarray:
    """
    :param im_path: JPEG image path
    :return: DCT_coef (Y,Cb,Cr)
    Code derived from https://github.com/mjkwon2021/CAT-Net.git.
    """
    extension = (image_path[-4:]).lower()
    assert extension == ".jpg" or extension == ".jpeg"

    jpeg = jpegio.read(str(image_path))
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

    DCT_coef = np.empty((num_channels, *jpeg.coef_arrays[0].shape))

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

        DCT_coef[i] = channel_coefficients

    return DCT_coef


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
        plot_multiple_images(
            self.images, self.names, ncols=ncols, title=title, save_path=save_path
        )


if __name__ == "__main__":
    test_base = Data.from_path(IMG_FOLDER_PATH)
    test_base.show(title="Prueba")
