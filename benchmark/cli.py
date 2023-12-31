import typer
from model import Benchmark

from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.datasets.registry import DatasetName
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.methods.registry import MethodName
from photoholmes.metrics.metric_factory import MetricFactory

# TODO: add a command to list the available methods, datasets and metrics
# TODO: add documentation for the CLI
app = typer.Typer()


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

    # Load metrics
    metrics_list = metrics.split()
    metrics_objects = MetricFactory.load(metrics_list)

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


if __name__ == "__main__":
    app()
