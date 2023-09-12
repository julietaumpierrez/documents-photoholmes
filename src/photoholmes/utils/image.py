import os
from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

IMG_FOLDER_PATH = "test_images/images/"


def plot_multiple_images(
    images, titles=None, ncols=4, title: str = None, save_path=None
):
    """D"""
    N = len(images)
    nrows = np.ceil(N / ncols).astype(int)
    fig, ax = plt.subplots(nrows, ncols)
    if titles is None:
        titles = [None * len(images)]
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
    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print("Figure saved at:", save_path)
    plt.show()

def read_mask(mask_path):
    '''Returns mask as a boolean image, from a mask path
    '''
    mask = cv.imread(mask_path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    return mask > mask.max()/2


@dataclass
class ImFile:
    name: str
    img: str
    mask: str = None

    @classmethod
    def from_path(cls, image_path: str, mask_path: str = None):
        """Initializes image from a given image_path, and optionally a mask path containing forgery ground truth."""
        name = image_path.split("/")[-1]
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path) if mask_path is not None else None
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
    def __init__(self, imfiles: list[ImFile], name: str = None) -> None:
        self.imfiles = imfiles
        self.name = name

    @property
    def images(self) -> list[np.ndarray]:
        return [imfile.img for imfile in self.imfiles]

    @property
    def masks(self) -> list[np.ndarray]:
        return [imfile.mask for imfile in self.imfiles]

    @property
    def names(self) -> list[np.ndarray]:
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
    def from_path(cls, img_folder_path: str, mask_folder_path: str = None):
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

    def show(self, ncols=4, title: str = None, save_path=None):
        """Displays the dataset"""
        plot_multiple_images(
            self.images, self.names, ncols=ncols, title=title, save_path=save_path
        )


if __name__ == "__main__":
    test_base = Data.from_path(IMG_FOLDER_PATH)
    test_base.show(title="Prueba")
