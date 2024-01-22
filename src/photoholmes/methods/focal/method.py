from typing import Any, Dict, List, Literal, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_kmeans import KMeans
from torchvision.transforms.functional import resize

from .utils import load_weights


class Focal(nn.Module):
    def __init__(
        self,
        net_list: List[Literal["HRNet", "ViT"]],
        weights: List[Union[str, Dict[str, Any]]],
    ):
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

            self.clustering = KMeans()

    def forward(self, x):
        Fo = self.network_list[0](x)
        Fo = Fo.permute(0, 2, 3, 1)

        B, H, W, C = Fo.shape
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

    def predict(self, image: torch.Tensor):
        _, im_H, im_W = image.shape

        # FIXME don't want to add this as a preprocessing step
        # since it shouldnt be there in the first place.
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

        print(Lo.shape, im_H, im_W)
        return resize(Lo.unsqueeze(0), [im_H, im_W]).squeeze(0)
