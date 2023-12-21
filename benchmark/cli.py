import argparse

from model import Benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run the Benchmark for image tampering detection."
    )
    parser.add_argument(
        "--method_name", type=str, required=True, help="Name of the method to use."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--tampered_only", action="store_true", help="Process tampered images only."
    )
    parser.add_argument(
        "--metrics", nargs="+", type=str, required=True, help="List of metrics to use."
    )
    parser.add_argument("--save_output", action="store_true", help="Save the output.")
    parser.add_argument(
        "--output_path", type=str, default="output/", help="Path to save the outputs."
    )

    args = parser.parse_args()

    benchmark = Benchmark(
        method_name=args.method_name,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        tampered_only=args.tampered_only,
        metrics_names=args.metrics,
        save_output=args.save_output,
        output_path=args.output_path,
    )

    benchmark.run()


if __name__ == "__main__":
    main()
