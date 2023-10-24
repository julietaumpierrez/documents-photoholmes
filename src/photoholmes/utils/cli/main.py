import logging
import os
from typing import Optional

import typer
from matplotlib import pyplot as plt
from typing_extensions import Annotated

from photoholmes.models.method_factory import MethodFactory, MethodName
from photoholmes.utils.image import ImFile

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
):
    if config is None:
        logger.warning("No config file was provided, using default configs.")

    model = MethodFactory.load(method, config)

    image = ImFile.open(str(image_path))

    mask = model.predict(image.img)

    plt.imshow(mask)
    if out_path is None:
        os.makedirs("out", exist_ok=True)
        out_path = "out/" + image_path.split("/")[-1]

    plt.savefig(out_path)


def cli():
    app()
