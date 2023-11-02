# EXIF as Language: Learning Cross-Modal Associations Between Images and Camera Metadata

This is the implementation of the [EXIF as Language](https://arxiv.org/pdf/2301.04647.pdf) paper. The original implementation could be found [here](https://github.com/hellomuffin/exif-as-language).

The code contained in this library was derived from [the original implementation](https://github.com/hellomuffin/exif-as-language) 

## Description

An image file contains not only the pixel values, but also a lot of extra-metadata that accompanies the image taken: camera model, exposure time, focal length, jpeg quantization details, etc... In this method the content of the image is contrasted with the exif information to detect any inconsistencies between what is "said" about the image and what the image is.

## Full overview

The method consist of training both an image and text encoder through contrastive learning, obtaining a single, cross-modal embedding space. The paper draws inspiration from openai's CLIP, hanging out the natural language for the EXIF information concatenated as a string. 

The result of this training scheme are two encoder, one image and text, that work in the same embedding space. In other words, patches from the same image should be close in the embedding space, while patches from  images that have different EXIF information shouldn't be close. This allows us to use the image embedder to detect images that have been spliced. If patches taken from the same image cluster in two or more regions of the embedding space, that means that the image is a splicing of images that share different EXIF data

## Usage

Add later usage of method 

## Results on benchmarking dataset

Add results of all metrics in our own benchmarking dataset

## Results on common datasets

Add results on common datasets 

## Citation

```tex
@article{zheng2023exif,
  title={EXIF as Language: Learning Cross-Modal Associations Between Images and Camera Metadata},
  author={Zheng, Chenhao and Shrivastava, Ayush and Owens, Andrew},
  journal={arXiv preprint arXiv:2301.04647},
  year={2023}
}
```