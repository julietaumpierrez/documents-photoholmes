import os
from pathlib import Path

import typer

huge_images = [
    "40574_0",
    "40574_1",
    "40574_2",
    "45988_1",
    "56635_0",
    "56635_1",
    "56635_2",
    "56785_0",
    "56785_1",
    "56785_2",
    "62981_0",
    "62981_1",
    "62981_2",
    "66943_0",
    "66943_1",
    "66943_2",
]

authentic_huge_images = [
    "62206",
    "47254",
    "62981",
    "40574",
    "66943",
    "56785",
    "56635",
]


app = typer.Typer()


@app.command()
def remove_images(
    data_dir: Path = typer.Argument(
        ..., exists=True, help="Path to AUTOSPLICE dataset."
    )
):
    for image in huge_images:
        os.remove(data_dir / "Forged_JPEG100" / f"{image}.jpg")
        os.remove(data_dir / "Forged_JPEG90" / f"{image}.jpg")
        os.remove(data_dir / "Forged_JPEG75" / f"{image}.jpg")

    for image in authentic_huge_images:
        os.remove(data_dir / "Authentic" / f"{image}.jpg")


if __name__ == "__main__":
    app()
