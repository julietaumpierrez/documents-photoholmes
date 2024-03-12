import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from photoholmes.methods import MethodRegistry

logging.basicConfig()

app = typer.Typer()


@app.command(name="run", help="Run a method on an image")
def run_method_cli(
    method: Annotated[
        MethodRegistry,
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


@app.command(name="adapt_weights", help="Adapt weights for a photoholmes method")
def run_adapt_weights(method: MethodRegistry, weights_path: str, out_path: str):
    if method == MethodRegistry.EXIF_AS_LANGUAGE:
        from photoholmes.methods.exif_as_language.prune_original_weights import (
            prune_original_weights,
        )

        prune_original_weights(weights_path, out_path)
    elif method == MethodRegistry:
        pass
    else:
        logging.info("No adaptation needed for this method.")


@app.command(name="download_weights", help="Automatic weight download for a method")
def run_download_weights(
    method: MethodRegistry,
    weight_folder_path: Annotated[
        Path, typer.Argument(help="Path to weight folder.")
    ] = Path("weights"),
):
    if method == MethodRegistry.PSCCNET:
        from .download_weights import download_focal_weights

        download_focal_weights(weight_folder_path / "psccnet")
    elif method == MethodRegistry.EXIF_AS_LANGUAGE:
        from .download_weights import download_exif_weights

        download_exif_weights(weight_folder_path / "exif_as_language")
    elif method == MethodRegistry.ADAPTIVE_CFA_NET:
        from .download_weights import download_adaptive_cfa_net_weights

        download_adaptive_cfa_net_weights(weight_folder_path / "adaptive_cfa_net")
    else:
        logging.info("No weights download for this method.")


@app.command(name="health", help="test the cli is working.")
def test():
    print("CLI is working!")


def cli():
    app()
