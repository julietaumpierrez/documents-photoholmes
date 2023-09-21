import os
from dataclasses import dataclass
from typing import Optional, Union

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct

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


def read_as_jpeg(image_path: str) -> np.ndarray:
    """Reads image from path and returns DCT coefficient matrix for each channel.
    If image is in jpeg format, it decodes the DCT stream and returns it.
    Otherwise, the DCT stream is calculated."""
    if image_path[-4:] == ".jpg" or image_path[-5:] == ".jpeg":
        return _read_jpeg(image_path)
    else:
        raise NotImplementedError()


def _read_jpeg(image_path: str) -> np.ndarray:
    """
    :param im_path: JPEG image path
    :return: DCT_coef (Y,Cb,Cr)
    """
    assert image_path[-4:] == ".jpg" or image_path[-5:] == ".jpeg"

    jpeg = jpegio.read(str(image_path))
    num_channels = len(jpeg.coef_arrays)

    # determine which axes to up-sample
    # ci = jpeg.
    ci = jpeg.comp_info
    need_scale = [
        [ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)
    ]
    if num_channels == 3:
        if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
            need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
        if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
            need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
    else:
        need_scale[0][0] = 2
        need_scale[0][1] = 2

    # up-sample DCT coefficients to match image size
    DCT_coef = np.empty((num_channels, *jpeg.coef_arrays[0].shape))

    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        coef_view = (
            jpeg.coef_arrays[i].reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        )
        # case 1: row scale (O) and col scale (O)
        if need_scale[i][0] == 1 and need_scale[i][1] == 1:
            out_arr = np.zeros((r * 2, c * 2))
            out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(
                0, 2, 1, 3
            )
            out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

        # case 2: row scale (O) and col scale (X)
        elif need_scale[i][0] == 1 and need_scale[i][1] == 2:
            out_arr = np.zeros((r * 2, c))
            # DCT_coef.append(out_arr)
            DCT_coef[i] = out_arr
            out_view = out_arr.reshape(r * 2 // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, :, :, :] = coef_view[:, :, :, :]
            out_view[1::2, :, :, :] = coef_view[:, :, :, :]

        # case 3: row scale (X) and col scale (O)
        elif need_scale[i][0] == 2 and need_scale[i][1] == 1:
            out_arr = np.zeros((r, c * 2))
            out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, ::2, :, :] = coef_view[:, :, :, :]
            out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

        # case 4: row scale (X) and col scale (X)
        elif need_scale[i][0] == 2 and need_scale[i][1] == 2:
            out_arr = np.zeros((r, c))
            out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, :, :, :] = coef_view[:, :, :, :]

        else:
            raise KeyError("Something wrong here.")

        DCT_coef[i] = out_arr

    return DCT_coef


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
