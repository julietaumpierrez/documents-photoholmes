import logging
import os
from typing import Optional

import typer
from matplotlib import pyplot as plt
from typing_extensions import Annotated

from photoholmes.models.base import BaseTorchMethod
from photoholmes.models.method_factory import MethodFactory, MethodName
from photoholmes.utils.image import ImFile, read_jpeg_data

app = typer.Typer()

logger = logging.getLogger("cli")
logger.setLevel(logging.WARNING)


@app.command(name="test", help="test the cli is working.")
def test():
    print("test")


@app.command(name="run", help="Run a method on a image.")
def run_model(
    method: Annotated[
        MethodName,
        typer.Argument(help="Method to run the image through.", case_sensitive=False),
    ],
    image_path: str,
    out_path: Optional[str] = None,
    config: Annotated[
        Optional[str],
        typer.Option(
            help="Path to '.yaml' config file. If None, default configs will be used.",
        ),
    ] = None,
    device: Optional[str] = None,
    num_dct_channels: Optional[int] = 1,
):
    if config is None:
        logger.warning("No config file was provided, using default configs.")

    model, preprocess = MethodFactory.load(method, config)

    if isinstance(model, BaseTorchMethod):
        model.to(device)

    image = ImFile.open(str(image_path)).img
    if image_path.split(".")[-1] in ["jpg", "jpeg"]:
        dct_channels, qtables = read_jpeg_data(image_path, num_dct_channels)
        x = preprocess(image=image, dct_coefficients=dct_channels, qtables=qtables)
    else:
        x = preprocess(image=image)

    mask = model.predict(**x)

    plt.imshow(mask)
    if out_path is None:
        os.makedirs("out", exist_ok=True)
        out_path = "out/" + image_path.split("/")[-1]

    plt.savefig(out_path)


def cli():
    app()
