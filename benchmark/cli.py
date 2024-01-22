from typing import List

import typer

from photoholmes.benchmark.model import Benchmark
from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.datasets.registry import DatasetName
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.methods.registry import MethodName
from photoholmes.metrics.metric_factory import MetricFactory
from photoholmes.utils.generic import load_yaml

# TODO: add a command to list the available methods, datasets and metrics
# TODO: add documentation for the CLI
app = typer.Typer()


def run_benchmark(
    method_name: MethodName,
    method_config: str | dict,
    dataset_name: DatasetName,
    dataset_path: str,
    metrics: List[str],
    tampered_only: bool = False,
    save_output: bool = False,
    output_path: str = "output/",
    device: str = "cpu",
):
    # Load method and preprocessing
    method, preprocessing = MethodFactory.load(
        method_name=method_name, config=method_config, device=device
    )

    # Load dataset
    dataset = DatasetFactory.load(
        dataset_name=dataset_name,
        dataset_dir=dataset_path,
        tampered_only=tampered_only,
        transform=preprocessing,
    )

    metrics_objects = MetricFactory.load(metrics)

    # Create Benchmark
    benchmark = Benchmark(
        save_output=save_output,
        output_path=output_path,
        device=device,
    )

    # Run Benchmark
    benchmark.run(
        method=method,
        dataset=dataset,
        metrics=metrics_objects,
    )


@app.command()
def main(
    method_name: MethodName = typer.Option(..., help="Name of the method to use."),
    method_config: str = typer.Option(
        None, help="Path to the configuration file for the method."
    ),
    dataset_name: DatasetName = typer.Option(..., help="Name of the dataset."),
    dataset_path: str = typer.Option(..., help="Path to the dataset."),
    metrics: str = typer.Option(
        ..., "--metrics", help="Space-separated list of metrics to use."
    ),
    tampered_only: bool = typer.Option(False, help="Process tampered images only."),
    save_output: bool = typer.Option(False, help="Save the output."),
    output_path: str = typer.Option("output/", help="Path to save the outputs."),
    device: str = typer.Option("cpu", help="Device to use."),
):
    """
    Run the Benchmark for image tampering detection.
    """

    # Load metrics
    metrics_list = metrics.split()
    run_benchmark(
        method_name=method_name,
        method_config=method_config,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        metrics=metrics_list,
        tampered_only=tampered_only,
        save_output=save_output,
        output_path=output_path,
        device=device,
    )


@app.command("from_config")
def run_from_config(
    config_path: str = typer.Option(..., help="Path to the configuration file.")
):
    config = load_yaml(config_path)
    run_benchmark(**config)


if __name__ == "__main__":
    app()
