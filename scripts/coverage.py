import os
from pathlib import Path

import typer

mismatched_images = ["41t", "57t", "61t", "95t", "56t", "55t", "59t", "58t", "48t"]


app = typer.Typer()


@app.command()
def remove_images(
    data_dir: Path = typer.Argument(..., exists=True, help="Path to COVERAGE dataset.")
):
    for image in mismatched_images:
        os.remove(data_dir / "image" / f"{image}.tif")


if __name__ == "__main__":
    app()
