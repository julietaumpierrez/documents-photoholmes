import logging
import os
from typing import Optional

import torch
from matplotlib import pyplot as plt

from photoholmes.methods.base import BaseTorchMethod
from photoholmes.methods.method_factory import MethodFactory, MethodName
from photoholmes.utils.image import ImFile, read_jpeg_data

log = logging.getLogger("cli.run_method")


def run_method(
    method: MethodName,
    image_path: str,
    out_path: Optional[str] = None,
    config: Optional[str] = None,
    device: Optional[str] = None,
    num_dct_channels: Optional[int] = 1,
    all_qtables: bool = False,
):
    if config is None:
        log.warning(
            "No config file was provided, using default configs. Method using "
            "pretrained weights will not work unless the path to the weights is "
            "provided."
        )

    model, preprocess = MethodFactory.load(method, config)

    if isinstance(model, BaseTorchMethod):
        if device is None and torch.cuda.is_available():
            log.info("Cuda detected.")
            model.to("cuda")
        elif device is not None:
            model.to(device)

    image = ImFile.open(str(image_path)).img
    dct_channels, qtables = read_jpeg_data(image_path, num_dct_channels, all_qtables)
    x = preprocess(image=image, dct_coefficients=dct_channels, qtables=qtables)

    print(f"Running {method.value}")
    mask = model.predict(**x)

    if len(mask.shape) > 2:
        mask = mask[0]

    plt.imshow(mask)
    if out_path is None:
        os.makedirs("out", exist_ok=True)
        out_path = f"out/{method.value}_{image_path.split('/')[-1]}"

    print(f"Saving mask to {out_path}")
    plt.savefig(out_path)
