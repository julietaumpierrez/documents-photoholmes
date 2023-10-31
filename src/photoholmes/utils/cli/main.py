import logging

import typer

from .run_method import run_method

logging.basicConfig()

app = typer.Typer()

app.command(name="run", help="Run a method on an image")(run_method)


@app.command(name="benchmark", help="test the cli is working.")
def test():
    print("test")


def cli():
    app()
