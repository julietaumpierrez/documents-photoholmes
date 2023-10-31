import logging
from typing import Optional

import typer
from typing_extensions import Annotated

from photoholmes.models.registry import MethodName

logging.basicConfig()

app = typer.Typer()


@app.command(name="run", help="Run a method on an image")
def run_method_cli(
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
    from .run_method import run_method

    run_method(method, image_path, out_path, config, device, num_dct_channels)


@app.command(name="benchmark", help="test the cli is working.")
def test():
    print("test")


def cli():
    app()
