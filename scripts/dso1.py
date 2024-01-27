import os
from pathlib import Path

import typer

mismatched_images = [
    "splicing-38",
    "splicing-44",
    "splicing-47",
    "splicing-42",
    "splicing-43",
]


app = typer.Typer()


@app.command()
def remove_images(
    data_dir: Path = typer.Argument(..., exists=True, help="Path to DSO1 dataset.")
):
    for image in mismatched_images:
        os.remove(data_dir / "DSO-1" / f"{image}.png")


if __name__ == "__main__":
    app()
