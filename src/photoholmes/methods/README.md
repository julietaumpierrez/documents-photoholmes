# Methods Module

## Overview
This module provides a collection of methods that can be used by themselves to make predictions on suspicious images and that can be used with the benchmark module to check their performance with the available datasets.


## Available Methods

In the first version of the PhotoHolmes library 10 methods are available:

- Adaptive CFA Net: An Adaptive Neural Network for Unsupervised Mosaic Consistency Analysis in Image Forensics. 
- CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing.
- DQ: Fast, Automatic, and Fine-Grained Tampered JPEG Image Detection via DCT Coefficient Analysis.
- EXIF as Language: Learning Cross-Modal Associations Between Images and Camera Metadata.
- FOCAL: Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering.
- Noisesniffer: Image forgery detection based on noise inspection: analysis and refinement of the Noisesniffer method.
- PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization.
- Splicebuster: a new blind image splicing detector.
- TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization.
- ZERO: A Local JPEG Grid Origin Detector Based on the Number of DCT Zeros and its Applications in Image Forensics.

For more information regarding the nature of each method please refer to their corresponding README.

# Structure

Methods in the PhotoHolmes library consist of at least these parts:
- method.py file: Contains the class that inherits from the BaseMethod class for non deep learning based methods or that inherits from the BaseTorchMethod class when they are deep learning based. This Class has at least three methods: 
    - __init__: that initializates the class.
    - predict: that given an image gives the original output of the method
    - benchmark: that given an image gives benchmarks the method by returning a BenchmarkOutput that is a dictionary containing the outputs that correspond. The accpeted outputs are: heatmap, mask and detection. 
- preprocessing.py: Contains the preprocessing pipeline needed for each method.
- config.yaml: YAML file that contains the example config for each method with default parameters.


## Method Factory

The `MethodFactory` class provides a way of loading the method and the correspondin preprocessing

It returns a Tuple containing the method object and the corresponding preprocessing pipeline.

## Examples of Use

Here are some examples of how to use the methods in this module:

### Importing the method directly:

You can easily use the chosen method by importing the method directly from the PhotoHolmes library.
If the method is not deep learning based it will look like this:
```python
from photoholmes.methods.chosen_method import Method, method_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = method_preprocessing(**image_data)

# Declare the method
method = Method()

# Use predict to get the final result
output = method.predict(**input)
```

If the method is deep learning based it will look like this:

```python
from photoholmes.methods.chosen_method import Method, method_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = method_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
path_to_weights = "path_to_weights"
method = Method(
    weights= path_to_weights
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```
Given the fact that these are deep learning based methods you will need the corresponding weights. For information on where to find them please refer to the method README.

Please be aware that some methods might need more information than just the image as input, please refer to each method's README or documentation in order to see how the input has to look like.

### Using the MethodFactory:

```python
# Import the MethodFactory

from photoholmes.methods.factory import MethodFactory

# Use the MethodFactory to import the method and preprocessing

method_name = "method_name"
config_path = "path_to_config"

method, preprocess = MethodFactory.load(method_name,config_path)

# Load an image

from photoholmes.utils.image import read_image

image_path = "image_path"
img = read_image(image_path)

# Use the preprocess and then do the prediction

inputs = preprocess(img)
out = method.predict(**inputs)
```
## Benchamarked results

With the PhotoHolmes library we reported the perfromance of all mentioned methods in different datasets and with different metrics. 
### Localization Performance

### Detection Performance

## Contribute: Adding a new method
1. Create the folder corresponding to the new method.
2. Create and fill all of the corresponding files described in the Structure section. Please be aware that some methods might need more files just as utils.py or postprocessing.py.
3. Add the method to the registry and to the factory.
4. Fill out the README and don't forget to include links to the weights if its a deep learning based method.

Make a pull request to the repository with the new method following the instructions of the [CONTRIBUTING.md](../CONTRIBUTING.md) file.