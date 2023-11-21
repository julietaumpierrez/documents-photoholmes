import logging
from math import floor
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PatchedImage:
    def __init__(
        self,
        data: torch.Tensor,
        patch_size: int,
        stride: Optional[int] = None,
        num_per_dim: Optional[int] = None,
    ):
        """Representation of an image that is sliced into patches

        Parameters
        ----------
        data: torch.Tensor
            - [C, H, W]
        patch_size: int
        stride: int | None
            - If None, it's calculated using num_per_dim
        num_per_dim: int | None
            - If stride is None, calculates it as as (max(height, widht) - patch_size) / self.stride
            - if None,
        """
        self.data = data.float()

        # Initialize image attributes
        self.patch_size = patch_size

        self.shape = data.shape
        _, height, width = data.shape

        # Compute patch stride on image
        if stride is not None:
            if num_per_dim is not None:
                logger.warn("Both `stride` and `num_per_dim` where set, using stride")
            self.stride = stride
        elif num_per_dim is not None:
            self.stride = (max(height, width) - self.patch_size) // num_per_dim
        else:
            self.stride = patch_size

        # Compute total number of patches along height and width dimension
        self.max_h_idx = 1 + floor((height - self.patch_size) / self.stride)
        self.max_w_idx = 1 + floor((width - self.patch_size) / self.stride)

    def get_patch(self, h_idx: int, w_idx: int) -> torch.Tensor:
        """Get a patch from the image

        Parameters
        ----------
        h_idx : int
        w_idx : int

        Returns
        -------
        torch.Tensor
            [C, patch_size, patch_size]
        """
        h_coord = h_idx * self.stride
        w_coord = w_idx * self.stride

        return self.data[
            :, h_coord : h_coord + self.patch_size, w_coord : w_coord + self.patch_size
        ]

    def get_patch_map(self, h_idx: int, w_idx: int) -> torch.Tensor:
        """Get a binary mask for the patch.

        Parameters
        ----------
        h_idx : int
        w_idx : int

        Returns
        -------
        torch.ByteTensor
            [H, W], values of {0, 1}
        """
        h_coord = h_idx * self.stride
        w_coord = w_idx * self.stride

        _, height, width = self.shape

        binary_map = torch.zeros(height, width, dtype=torch.bool)
        binary_map[
            h_coord : h_coord + self.patch_size, w_coord : w_coord + self.patch_size
        ] = True

        return binary_map

    def get_patches(self, idxs: NDArray) -> torch.Tensor:
        """Get patches from image given its indices

        Parameters
        ----------
        idxs : torch.Tensor
            [n_patches, 2], [n_patches, (h_idx, w_idx)]

        Returns
        -------
        torch.Tensor
            [n_patches, _, patch_size, patch_size]
        """
        n_patches = idxs.shape[0]
        patches = torch.zeros(
            n_patches,
            self.shape[0],
            self.patch_size,
            self.patch_size,
            device=self.data.device,
        )

        # FIXME Any way to vectorize this?
        # https://discuss.pytorch.org/t/advanced-fancy-indexing-across-batches/103445
        for i, idx in enumerate(idxs):
            h_idx, w_idx = idx

            patches[i] = self.get_patch(h_idx, w_idx)

        return patches

    def get_patch_maps(self, idxs: NDArray) -> torch.Tensor:
        n_patches = idxs.shape[0]
        _, height, width = self.shape

        maps = torch.zeros(n_patches, height, width, dtype=torch.bool)

        for i, idx in enumerate(idxs):
            h_idx, w_idx = idx
            maps[i] = self.get_patch_map(h_idx, w_idx)

        return maps

    def patches_gen(self, batch_size=32):
        """Generator for all patches in an image, in raster scan order

        Parameters
        ----------
        batch_size : int, optional
            Number of patches in each iteration, by default 32

        Returns
        -------
        torch.Tensor
            [batch_size, _, patch_size, patch_size]
        """
        count = 0

        # Initialize indices / coords of all patches, [n_patches, 2]
        h_idxs = np.arange(self.max_h_idx)
        w_idxs = np.arange(self.max_w_idx)
        idxs = np.stack(np.meshgrid(h_idxs, w_idxs)).transpose(0, 2, 1).reshape(2, -1).T

        n_patches = len(idxs)

        while count * batch_size < n_patches:
            # Yield a batch of patches
            patches = self.get_patches(
                idxs[count * batch_size : (count + 1) * batch_size]
            )
            yield patches
            count += 1

    def patch_maps_gen(self, batch_size=32):
        count = 0

        # Initialize indices / coords of all patches, [n_patches, 2]
        h_idxs = np.arange(self.max_h_idx)
        w_idxs = np.arange(self.max_w_idx)
        idxs = np.stack(np.meshgrid(h_idxs, w_idxs)).transpose(0, 2, 1).reshape(2, -1).T

        n_patches = len(idxs)

        while count * batch_size < n_patches:
            patches = self.get_patch_maps(
                idxs[count * batch_size : (count + 1) * batch_size]
            )
            yield patches
            count += 1

    def pred_idxs_gen(self, batch_size=32):
        """Generator for all prediction map indices.

        Parameters
        ----------
        batch_size : int, optional
            Number of indices in each iteration, by default 32

        Returns
        -------
        torch.Tensor
            [batch_size, 4]
        """
        idxs = (
            np.mgrid[
                0 : self.max_h_idx,
                0 : self.max_w_idx,
                0 : self.max_h_idx,
                0 : self.max_w_idx,
            ]
            .reshape((4, -1))
            .T
        )

        count = 0
        while count * batch_size < len(idxs):
            yield idxs[count * batch_size : (count + 1) * batch_size]
            count += 1

    def idxs_gen(self, batch_size=32):
        """Generator for indices of the image_patches.

        Parameters
        ----------
        batch_size : int, optional
            Number of indices in each iteration, by default 32

        Returns
        -------
        torch.Tensor
            [batch_size, 4]
        """
        idxs = (
            np.mgrid[
                0 : self.max_h_idx,
                0 : self.max_w_idx,
            ]
            .reshape((2, -1))
            .T
        )

        count = 0
        while count * batch_size < len(idxs):
            yield idxs[count * batch_size : (count + 1) * batch_size]
            count += 1
