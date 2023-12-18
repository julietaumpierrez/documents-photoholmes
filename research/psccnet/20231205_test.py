import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from photoholmes.methods.method_factory import MethodFactory
from photoholmes.methods.psccnet.method import PSCCNet
from photoholmes.methods.psccnet.their_docs.load_vdata import TestData
from photoholmes.utils.image import plot, plot_multiple, read_image

IM_PATH = "data/copymove2.png"
DEVICE = "cpu"


def migration_test():
    method, preprocess = MethodFactory.load(
        "psccnet", "src/photoholmes/methods/psccnet/config.yaml"
    )
    image = read_image(IM_PATH).float()[None, :, :, :] / 255
    image = image.to(DEVICE)

    y = method.predict(image)

    feat = y[
        "feat"
    ]  # Para esto hay que poner que el predict devuelva el feat. Esto no va a correr cuando deje de estar debuuggeando
    pred_mask = y["mask"]

    return feat, pred_mask


if __name__ == "__main__":
    im = read_image(IM_PATH)[None, :, :, :]

    feat, pred_mask = migration_test()

    true_mk_pred = torch.load("data/debug/copymove2.png.pth")
    assert true_mk_pred[0].size() == feat[0].size()
    assert (true_mk_pred[0] == feat[0]).all()

    plot_multiple([im[0], pred_mask[0]], title="Imagen y máscara predecida")
