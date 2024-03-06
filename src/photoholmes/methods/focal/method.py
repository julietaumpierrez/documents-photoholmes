from typing import Any, Dict, List, Literal, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_kmeans import KMeans
from torchvision.transforms.functional import resize

from photoholmes.methods.base import BaseTorchMethod, BenchmarkOutput

from .utils import load_weights


class Focal(BaseTorchMethod):
    """
    Implementation of Focal [H. Wu and Y. Chen and J. Zhou, 2023].

    Focal is an end to end neural network.
    """

    def __init__(
        self,
        net_list: List[Literal["HRNet", "ViT"]],
        weights: List[Union[str, Dict[str, Any]]],
        device: str = "cpu",
    ):
        """
        Attributes:
            net_list (List[str]): list of networks to be used in the ensemble.
            weights (List[str | dict]): list of weights for the networks in the
                ensemble.
            device (str): device to run the model on.
        """
        super().__init__()

        self.network_list = nn.ModuleList()

        for net_name, w in zip(net_list, weights):
            if net_name == "HRNet":
                from .models.hrnet import HRNet

                net = HRNet()
                load_weights(net, w)

                self.network_list.append(net)

            elif net_name == "ViT":
                from .models.vit import ImageEncoderViT

                net = ImageEncoderViT()
                load_weights(net, w)

                self.network_list.append(net)

            self.clustering = KMeans(verbose=False)
        self.to_device(device)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): input image of shape (B, C, H, W)
        """
        Fo = self.network_list[0](x)
        Fo = Fo.permute(0, 2, 3, 1)

        _, H, W, _ = Fo.shape
        Fo = F.normalize(Fo, dim=3)
        Fo_list = [Fo]

        for additional_net in self.network_list[1:]:
            Fo_add = additional_net(x)
            Fo_add = F.interpolate(Fo_add, (H, W))
            Fo_add = Fo_add.permute(0, 2, 3, 1)
            Fo_add = F.normalize(Fo_add, dim=3)
            Fo_list.append(Fo_add)

        Fo = torch.cat(Fo_list, dim=3)

        return Fo

    def predict(self, image: torch.Tensor):  # type: ignore[override]
        """
        Run a prediction over a preprocessed image. You can use the pipeline
        `focal_preprocessing` provied in `photoholmes.methods.focal.preprocessing`.

        Args:
            image (torch.Tensor): input image of shape (C, H, W)

        Returns:
            Tensor: binary mask of shape (H, W)
        """
        if len(image.shape) != 3:
            raise ValueError("Input image should be of shape (C, H, W)")
        _, im_H, im_W = image.shape

        # This operation destroys any traces of forgery the method might
        # exploit. This indicates the method is most likely overfitted to
        # the dataset.
        image = resize(image, [1024, 1024])

        with torch.no_grad():
            Fo = self.forward(image[None, :])
            _, W, H, _ = Fo.shape
            Fo = Fo.flatten(1, 2)

        result = self.clustering(x=Fo, k=2)

        Lo = result.labels
        if torch.sum(Lo) > torch.sum(1 - Lo):
            Lo = 1 - Lo
        Lo = Lo.view(H, W)
        mask = resize(Lo.unsqueeze(0), [im_H, im_W]).squeeze(0).float()

        return mask

    def benchmark(  # type: ignore[override]
        self, image: torch.Tensor
    ) -> BenchmarkOutput:
        """
        Wrapper of the `predict` method to be used in the benchmark pipeline.
        """
        mask = self.predict(image)
        return {"mask": mask, "heatmap": None, "detection": None}
