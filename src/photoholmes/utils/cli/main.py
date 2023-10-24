import logging
from pathlib import Path
from typing import Optional

import typer
from PIL.Image import Image
from typing_extensions import Annotated

from photoholmes.models.method_factory import MethodFactory, MethodName

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.command(name="test", help="test the cli is working.")
def test():
    print("test")


@app.command(name="run_method", help="Run a method on a image.")
def run_model(
    method: Annotated[
        MethodName,
        typer.Argument(help="Method to run the image through.", case_sensitive=False),
    ],
    image_path: Path,
    config: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to '.yaml' config file. If None, default configs will be used.",
        ),
    ] = None,
):
    if config is None:
        logger.warn(f"No config file was provided, using default configs.")
    print(method, image_path)


def cli():
    app()
