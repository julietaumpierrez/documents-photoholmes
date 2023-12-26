import typer
from model import Benchmark

from photoholmes.datasets.registry import DatasetName
from photoholmes.methods.registry import MethodName
from photoholmes.metrics.registry import MetricName

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
    metrics_list = metrics.split()
    metrics_names = [MetricName(metric) for metric in metrics_list]

    benchmark = Benchmark(
        method_name=method_name,
        method_config=method_config,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tampered_only=tampered_only,
        metrics_names=metrics_names,
        save_output=save_output,
        output_path=output_path,
        device=device,
    )

    benchmark.run()


if __name__ == "__main__":
    app()
