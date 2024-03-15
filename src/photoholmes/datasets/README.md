# Datasets Module

## Overview
This module provides a collection of datasets that can be used to test the performance of the methods in the methods module.
The different datasets are selected to cover a wide range of image manipulation techniques and to provide a good benchmark for the methods.

The datasets cover a wide range of forgery types as well as image formats, which we deemed important to benchmark the diverse array of included methods. Besides the original datasets, some of them have their social media versions which are also included in the library. In addition, we included a WebP version of the Korus dataset since, up to our knowledge, no forgery detection dataset features this increasingly popular format.

## Available Datasets

The following datasets are available in the PhotoHolmes library:
- Columbia: A dataset of spliced images.
- Coverage: A dataset of copy-move manipulated images.
- DSO-1: A dataset of spliced images.
- Korus: A dataset of spliced, copy-move, and object removal manipulated images.
- Casia 1.0: A dataset of spliced and copy-move manipulated images.
- AutoSplice: A dataset of generative inpainting manipulated images.
- Trace: A dataset of images with alterations to the acquisition pipeline.

The following table provides an overview of the datasets and their characteristics:

| Dataset | Types of Forgery | Nb. of Images (🔵 forged + 🟠 pristine) | Format | Social Media Version | WebP Version |
|---------|------------------|----------------------------------------|--------|----------------------|--------------|
| Columbia | Splicing | 363 (🔵180 + 🟠183) | TIF | ✅ | ❌ |
| Coverage | Copy-move | 200 (🔵100 + 🟠100) | TIF | ❌ | ❌ |
| DSO-1 | Splicing | 200 (🔵100 + 🟠100) | PNG | ✅ | ❌ |
| Korus | Splicing, copy-move, object removal | 440 (🔵220 + 🟠220) | TIF | ❌ | ✅ |
| Casia 1.0 | Splicing, copy-move | 1023 (🔵923 + 🟠100) | JPEG | ✅ | ❌ |
| AutoSplice | Generative inpainting | 5894 (🔵3621 + 🟠2273) | JPEG | ❌ | ❌ |
| Trace | Alterations to acquisition pipeline | 24000 (🔵24000 + 🟠0) | PNG | ❌ | ❌ |

### Columbia
This dataset contains spliced images, which are not realistic at all and could be easily detected by semantic evaluation. This means that just by looking at the image and considering the context, a person can identify the suspicious area. One could argue that detecting forgeries of this type does not add value to a method, as they can be easily identified by the human eye. However, the importance of this dataset lies not only in its popularity but also in the fact that it has its version through different social networks. With the correct metrics, it allows for the quantification of how well or poorly a method can generalize \textit{in the wild} forgeries, especially in the context of the different processing an image undergoes when uploaded to any social network. 

### Coverage
It is the most popular dataset for evaluating copy-move forgeries. The images in this dataset are uncompressed, and the pristine images consistently feature a repetition of a certain object. For the forged images, one of these objects is cut and pasted elsewhere, with the pasted object sometimes easily located and other times not. This dataset helps determine whether a method merely searches for similar parts within the image to detect a copy-move forgery or if it looks for inconsistencies in traces, such as the demosaicing grid.

### DSO-1
DSO-1 is a dataset that contains spliced images in which the subject used for the splicing are humans. At first glance, the splices are hard to catch, however most of the times, doing a semantic evaluation regarding the light shows which subject is spliced. This dataset is of PNG images and it has its version through different social networks.

### Korus
The Korus dataset is also named realistic tampering. As the title suggests, this dataset contains forgeries that are almost impossible to detect through semantic evaluation. It has uncompressed images containing splicing copy move and object removal. 

### Casia 1.0
This dataset contains both splicing and copy move forgeries which are not so easy to identify to the naked eye and are JPEG compressed. It also has its version through different social networks which allows the same analysis as Columbia on top of being spliced and copy move forgeries JPEG compressed.

### AutoSplice
This novel dataset is unique as it incorporates generative inpainting. Jia et al. introduce the utilization of to generate forged images guided by a text prompt. These images are JPEG compressed, and the dataset includes variations with three JPEG quality factors: 100, 90, and 75. This diversity facilitates the quantification of how well methods can handle varying degrees of JPEG compression.

### Trace
In Trace, the forged and pristine regions differ only in the traces left behind by the imaging pipeline. The concept involves selecting a raw image and processing it using two distinct imaging pipelines. The results are then merged, forming a single image with two areas, each corresponding to one of the two pipelines. The merging of these images is accomplished using a mask.

# Structure

## BaseDataset
The `BaseDataset` class takes care of loading the images data and the masks from the dataset.

Functionalities:
- It also takes care of preprocessing the images data if a preprocessing pipeline is provided.
- You can load the following image data from the datasets:
    - Image: The original image.
    - DCT Coefficients: The DCT coefficients of the image.
    - Q Tables: The Q tables of the image.
- You can choose to load tampered only images or tampered and pristine images.
- The name of the image is also retrieved from the path of the image. This is useful for the evaluation of the methods and saving the results.
- Mask binarization

## Custom Datasets
The datasets are structured in the following way:
- dataset.py file: Contains the class that inherits from the BaseDataset class. This class has at least two methods and declare two attributes:
    - _get_paths: that returns the paths to the images in the dataset.
    - _get_mask_path: that returns the path to the mask given an image path.
    - IMAGE_EXTENSION: the extension of the images in the dataset.
    - MASK_EXTENSION: the extension of the masks in the dataset.

The two methods are used to get the paths to the images and the masks in the dataset. As different datasets have different structures, these methods are implemented in each dataset class. The two attributes are used to show warning messages when jpeg data is requested from a dataset that does not have jpeg images or masks.
You can also override the `binarize_mask` method if you want to binarize the mask in a different way than the default one.


## Dataset Factory

The `DatasetFactory` class provides a way of loading the datasets. It has a method called `load` that takes the name of the dataset, the path to the dataset, an optional preprocessing pipeline, an optional flag to load only tampered images, and a parameter indicating which image data to load. It returns an instance of the dataset and the corresponding preprocessing pipeline.

## Examples of Use

Here are some examples of how to use the datasets:

### Importing the dataset directly:

You can easily use the chosen dataset by importing the dataset directly from the PhotoHolmes library.

```python
from photoholmes.datasets.columbia import ColumbiaDataset

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=True,
    load=["image"]
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
```

For loading the pristine images as well you can do the following:

```python
from photoholmes.datasets.columbia import ColumbiaDataset

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=False,
    load=["image"]
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
```

For iterating over the dataset you can do the following:

```python
from photoholmes.datasets.columbia import ColumbiaDataset

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=True,
    load=["image"]
)

# Iterate over the dataset
for data, mask, image_name in dataset:
    image = data["image"]
```

For loading the DCT coefficients and Q tables you can do the following:

```python
from photoholmes.datasets.columbia import ColumbiaDataset

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=True,
    load=["image", "dct_coefficients", "qtables"]
)

# Get the first image, the DCT coefficients and the Q tables
data, mask, image_name = dataset[0]
image = data["image"]
dct_coefficients = data["dct_coefficients"]
qtables = data["qtables"]
```

### Using the DatasetFactory:

You can also use the DatasetFactory to import the dataset. Here is an example of how to use the DatasetFactory with the Columbia dataset:

```python
# Import the DatasetFactory
from photoholmes.datasets.factory import DatasetFactory

# Use the DatasetFactory to import the dataset
dataset_name = "columbia"
dataset_path = "dataset_path"
preprocessing_pipeline = None
tampered_only = True
load = ["image"]

dataset, preprocess = DatasetFactory.load(
    dataset_name =dataset_name
    dataset_path = dataset_path,
    preprocessing_pipeline = preprocessing_pipeline,
    tampered_only = tampered_only,
    load=load,
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
```

### Using a preprocessing pipeline:

You can also use a preprocessing pipeline to preprocess the images before using them. Here is an example of how to use a preprocessing pipeline with the Columbia dataset:

```python
from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.methods.dq import dq_preprocessing

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=dq_preprocessing,
    tampered_only=True,
    load=["image"]
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
```

## Contribute: Adding a new dataset
1. Create a new file for the dataset in the datasets folder.
2. Create and fill all of the corresponding files described in the Structure section.
3. Add the dataset to the registry and to the factory.
4. Fill out the README and don't forget to include the characteristics of the dataset.
5. Make a pull request to the repository with the new dataset following the instructions of the [CONTRIBUTING.md](../CONTRIBUTING.md) file.

