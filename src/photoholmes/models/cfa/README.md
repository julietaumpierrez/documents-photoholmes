# "An Adaptive Neural Network for Unsupervised Mosaic Consistency Analysis in Image Forensics"

This is the implemenattion of the model presented in [[1]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bammey_An_Adaptive_Neural_Network_for_Unsupervised_Mosaic_Consistency_Analysis_in_CVPR_2020_paper.pdf).

The code contained in this library was derived from [the original implementation](https://github.com/qbammey/adaptive_cfa_forensics), making only minor changes to fit our project structure.


## Description

This research paper presents an innovative approach to automatically detect suspicious regions in potentially forged images. The method uses a Convolutional Neural Network (CNN) to identify inconsistencies in image mosaics, specifically targeting the artifacts left by demosaicing algorithms. Unlike many blind detection neural networks, this approach does not require labeled training data and can adapt to new, unseen data quickly.

## Full Overview

This research addresses the critical challenge of detecting image forgeries, focusing on the detection of demosaicing artifacts. Demosaicing is a key process in digital photography, where cameras use a Color Filter Array (CFA) to create color images, with the Bayer matrix being the most common. This process involves interpolating the full color image from pixels that are individually sampled in only one color, leaving unique artifacts. The study introduces a specialized Convolutional Neural Network (CNN) designed to detect these mosaic artifacts. Unlike traditional methods requiring extensive labeled datasets, this CNN can be trained on unlabelled, potentially forged images, showcasing an innovative approach in forensic image analysis.

To evaluate this method, the authors created a diverse benchmark database using the Dresden Image Database, processed with various demosaicing algorithms. This database comprises both authentic and forged images, where forgeries are created by splicing parts of images demosaiced differently. This setup allows for a detailed assessment of the network's capacity to detect inconsistencies in mosaic patterns indicative of forgery. The study demonstrates the network's effectiveness in detecting forgeries and its adaptability to different data types and compression formats, making a significant contribution to the field of image forensics by providing a robust, adaptable tool for unsupervised forgery detection.

# TODO: finish README
## Usage

Add later usage of method 

## Results on benchmarking dataset

Add results of all metrics in our own benchmarking dataset

## Results on common datasets

Add results on common datasets 

## Citation

```
@InProceedings{Bammey_2020_CVPR,
author = {Bammey, Quentin and Gioi, Rafael Grompone von and Morel, Jean-Michel},
title = {An Adaptive Neural Network for Unsupervised Mosaic Consistency Analysis in Image Forensics},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```