import logging
from typing import Optional

import typer
from typing_extensions import Annotated

from photoholmes.methods.registry import MethodName

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
    all_qtables: bool = False,
):
    from .run_method import run_method

    run_method(
        method, image_path, out_path, config, device, num_dct_channels, all_qtables
    )


@app.command(name="adapt-weights", help="Adapt weights for a photoholmes method")
def run_adapt_weights(method: MethodName, weights_path: str, out_path: str):
    match method:
        case MethodName.EXIF_AS_LANGUAGE:
            from photoholmes.methods.exif_as_language.prune_original_weights import (
                prune_original_weights,
            )

            prune_original_weights(weights_path, out_path)
        case _:
            logging.info("No adaptation needed for this method.")


@app.command(name="health", help="test the cli is working.")
def test():
    print("CLI is working!")


def cli():
    app()
