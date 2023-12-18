# PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization

This is the implemenattion of the PSCC-Net model presented [[1]](https://arxiv.org/abs/2103.10596).

The code contained in this library was derived from [the original implementation](https://github.com/proteus1991/PSCC-Net), making only minor changes to fit our project structure.

## Description

PSCC-Net is an end-to-end fully convolutional neural network. It consists of a neural network that using a coarse to fine approach returns a mask locating forgeries in the input image. The method also returns an answer to the detection problem by returning a label that indicates whether the image was manipulated or not.


## Full overview

The network is divided into two different steps, first the top down path is constituted by a backbone
called HRNetV2p-W18. The main goal of this part is compute features at different scales that serve as inputs to the same levels of the bottom up path. The features obtained at every level of the top down path are used as inputs to a detection head that indicates if the image is pristine or not.

Then, the authors use in every level Spatio-Channel Correlation Module (SCCM) that tries to lay hold of spatial and channel wise correlations. Here, a coarse to fine approach is used, it involves an increasingly more precise definition of the masks as its shown in the bottom up path. The full architecture is trained on synthetic dataset that includes splicing, removal, copy move and pristine images.

# TODO: finish README
## Usage

Add later usage of method 

## Results on benchmarking dataset

Add results of all metrics in our own benchmarking dataset

## Results on common datasets

Add results on common datasets 

## Citation

* [1] PSCC-Net
```tex
@article{liu2022pscc,
  title={PSCC-Net: Progressive spatio-channel correlation network for image manipulation detection and localization},
  author={Liu, Xiaohong and Liu, Yaojie and Chen, Jun and Liu, Xiaoming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
```
```
