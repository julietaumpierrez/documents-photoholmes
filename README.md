# PhotoHolmes

## Introduction

PhotoHolmes is an open-source _python_ library designed to run easily and benchmark forgery 
detection methods on digital images. The library includes an implementation of popular and 
state-of-the-art methods, datasets and evaluation metrics, all of which can be easily extended
by users with their own custom methods, datasets and metrics. The user can also evaluate 
methods via the command-line-interface (CLI) or the Benchmark class.

## Development setup

The code has been developed using Python 3.10.13. Create a virtual enviroment, either with conda or with pip. 
Activate the enviroment and install the library and required packages. For the ladder, there are two options.

### Install: Benchmarking and library use

If you only wish to use the library as a user, for bencharking or other uses, run:

```
pip install -e .
```
### Install: Develop

If you wish to develop on the library run:
```
pip install -e .[dev]
```

You must also install pre-commit hooks. Pre-commit runs check before a commit to ensure the code quality is being preserved. To install the git hooks, run:
```bash
pre-commit install
```

## Benchmarking

One of the main PhotoHolmes library features is the capacity to easily bechmark a `method` over a `dataset`, evaluating on a set of `metrics`. This means one can evaluate existing and custom methods uniformly and fairly, simplifying the process of method comparison.

This can be invoked in a simple manner by creating an instance of the `Benchmark` class and calling the `run` method, in the following way:

```python
from photoholmes.benchmark import Benchmark

benchmark = Benchmark(output_folder="output/folder/path")

random_method_results = benchmark.run(method=method, dataset=dataset, metrics=metrics)
```

Naturally, `method`, `dataset` and `metrics` must be instances of PhotoHolmes' `Method`, `Dataset` and `MetricCollection` respectively, which can be quicly instanced from classes implemented in PhotoHolmes by using the factory, as shown in the following example.

```python
from photoholmes.methods.factory import MethodFactory, MethodRegistry
from photoholmes.datasets.factory import DatasetFactory, DatasetRegistry
from photohoolmes.metrics.factory import MetricsFactory, MetricsRegsitry

method, method_preprocessing = MethodFactory.load(MethodRegistry.IMPLEMENTED_METHOD)
dataset = DatasetFactory.load(
    DatasetRegistry.DATASET,
    dataset_path="dataset/directory/path",
    load=["image"],
    preprocessing_pipeline=method_preprocessing,
)
metrics = MetricFactory.load([MetricsRegistry.AUROC, MetricsRegistry.IOU])
```

However, due to the library's easy extensibility, custom methods, datasets and metrics can be easily integrated into this framework. A good example that shows different ways of implementing objects and integrating custom ones can be found in the [benchmarking a method notebook](notebooks/benchmarking_a_method.ipynb). There is also more detailed documentation and examples for each module in the respective [methods](src/photoholmes/methods/README.md), [datasets](src/photoholmes/datasets/README.md) and [metrics](src/photoholmes/metrics/README.md) READMEs.

## `run` the Photolmes CLI

Benchmarking aside, in PhotoHolmes it is even quicker and more practical when it comes to evaluating implemented methods on single images. This can be done using the PhotoHolmes CLI with the command `run`, as follows:

```bash
photoholmes run zero --output-folder output/folder/path image/path
```

The CLI includes more useful commands, such as `download_weights` to easily download the weights of the implemented methods. Read the [CLI documentation](src/photoholmes/cli/README.md) for a better description on what it can do and how to use it.

## Licence


