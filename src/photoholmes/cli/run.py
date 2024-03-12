import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from matplotlib import pyplot as plt

from photoholmes.methods import MethodFactory, MethodRegistry
from photoholmes.methods.base import BaseTorchMethod
from photoholmes.utils.image import read_image, read_jpeg_data

logger = logging.getLogger("cli.run_method")


run_app = typer.Typer(name="run")


def run_method(
    method: MethodRegistry,
    image_path: str,
    out_path: Optional[str] = None,
    config: Optional[str] = None,
    device: Optional[str] = None,
    num_dct_channels: Optional[int] = 1,
    all_qtables: bool = False,
):
    if config is None:
        logger.warning(
            "No config file was provided, using default configs. Method using "
            "pretrained weights will not work unless the path to the weights is "
            "provided."
        )

    model, preprocess = MethodFactory.load(method, config)

    if isinstance(model, BaseTorchMethod):
        if device is None and torch.cuda.is_available():
            logger.info("Cuda detected.")
            model.to("cuda")
        elif device is not None:
            model.to(device)

    image = read_image(image_path)
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


@run_app.command("focal")
def run_focal(
    image_path: Annotated[Path, typer.Argument(help="Path to image to analyze.")],
    output_folder: Annotated[
        Optional[Path], typer.Option(help="Path to folder to solve outputs.")
    ] = None,
    vit_weights: Annotated[
        Optional[Path], typer.Option(help="Path to the ViT weights.")
    ] = None,
    hrnet_weights: Annotated[
        Optional[Path], typer.Option(help="Path to the HRNet weights.")
    ] = None,
):
    from photoholmes.methods.focal import Focal, focal_preprocessing
    from photoholmes.utils.image import read_image

    image = read_image(str(image_path))
    model_input = focal_preprocessing(image=image)

    if vit_weights is None:
        logger.info(
            "No ViT weights provided, using default path `weights/focal/VIT_weights.pth`."  # noqa: E501
        )
        vit_weights = Path("weights/focal/VIT_weights.pth")
        if not vit_weights.exists():
            logger.error(
                "ViT weights not found. Please provide the correct path, or run "
                "`photoholmes run download_weights focal` to download them."
            )
            return
    if vit_weights is None:
        logger.info(
            "No ViT weights provided, using default path `weights/focal/HRNET_weights.pth`."  # noqa: E501
        )
        vit_weights = Path("weights/focal/HRNET_weights.pth")
        if not vit_weights.exists():
            logger.error(
                "HRNet weights not found. Please provide the correct path, or run "
                "`photoholmes run download_weights focal` to download them."
            )
            return

    focal = Focal(weights={"ViT": str(vit_weights), "HRNet": str(hrnet_weights)})

    mask = focal.predict(**model_input)

    plt.imshow(mask.numpy())
    if output_folder is not None:
        os.makedirs(output_folder)

        plt.savefig(output_folder / f"{image_path.stem}_focal_mask.png")
        logger.info(
            f"Mask saved to {output_folder / f'{image_path.stem}_focal_mask.png'}"
        )
    elif output_folder is None:
        plt.show()

    return
