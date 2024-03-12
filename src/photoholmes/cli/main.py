import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

from photoholmes.methods import MethodRegistry

from .run import run_app

logging.basicConfig()
logger = logging.getLogger(__name__)

app = typer.Typer()
app.add_typer(run_app, name="run")


@app.command(name="adapt_weights", help="Adapt weights for a photoholmes method")
def run_adapt_weights(
    method: Annotated[
        MethodRegistry,
        typer.Argument(help="Method to run the image through.", case_sensitive=False),
    ],
    weights_path: str,
    out_path: str,
):
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
    method: Annotated[
        MethodRegistry,
        typer.Argument(help="method", case_sensitive=False),
    ],
    weight_folder_path: Annotated[
        Path, typer.Argument(help="Path to weight folder.")
    ] = Path("weights"),
):
    if method == MethodRegistry.PSCCNET:
        from .download_weights import download_psccnet_weights

        download_psccnet_weights(weight_folder_path / "psccnet")
    elif method == MethodRegistry.EXIF_AS_LANGUAGE:
        from .download_weights import download_exif_weights

        download_exif_weights(weight_folder_path / "exif_as_language")
    elif method == MethodRegistry.ADAPTIVE_CFA_NET:
        from .download_weights import download_adaptive_cfa_net_weights

        download_adaptive_cfa_net_weights(weight_folder_path / "adaptive_cfa_net")
    elif method == MethodRegistry.CATNET:
        from .download_weights import download_catnet_weights

        logger.warning(
            "CatNet weights are under a non-commercial license. See https://github.com/mjkwon2021/CAT-Net/tree/main?tab=readme-ov-file#licence for more information."  # noqa: E501
        )
        r = input("Press yes if you agree to the license [yes/no]: ")
        while r.lower() not in ["no", "n", "yes", "y"]:
            r = input("Press yes if you agree to the license [yes/no]: ")
        if r.lower() in ["no", "n"]:
            logger.warning("You must agree to the license to download the weights.")
            return
        download_catnet_weights(weight_folder_path / "catnet")
    elif method == MethodRegistry.TRUFOR:
        from .download_weights import download_trufor_weights

        logger.warning(
            "TruFor weights are under a non-commercial license. See https://github.com/grip-unina/TruFor/blob/main/test_docker/LICENSE.txt for more information."  # noqa: E501
        )
        r = input("Press yes if you agree to the license [yes/no]: ")
        while r.lower() not in ["no", "n", "yes", "y"]:
            r = input("Press yes if you agree to the license [yes/no]: ")
        if r.lower() in ["no", "n"]:
            logger.warning("You must agree to the license to download the weights.")
            return

        download_trufor_weights(weight_folder_path / "trufor")
    elif method == MethodRegistry.FOCAL:
        from .download_weights import download_focal_weights

        download_focal_weights(weight_folder_path / "focal")
    else:
        logging.info(
            "No weights available for this method. Check the method README "
            "for more information."
        )


@app.command(name="health", help="test the cli is working.")
def test():
    print("CLI is working!")


def cli():
    app()
