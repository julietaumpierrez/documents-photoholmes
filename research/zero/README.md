# ZERO: a Local JPEG Grid Origin Detector Based on the Number of DCT Zeros and its Applications in Image Forensics

This is the implementation of the [ZERO](https://www.ipol.im/pub/art/2021/390/article_lr.pdf) paper.

The code contained in this library was derived from the original implementation in C, available at the [IPOL website](https://www.ipol.im/pub/art/2021/390/?utm_source=doi).

## Description

The method detects JPEG compression as well as its grid origin. This method can be applied globally to identify a JPEG compression, and also locally to identify image forgeries when misaligned or missing JPEG grids are found. This allows image forensics to be applied, by identifying anomalies in the grid encountered locally with respect to the main grid detected.

## Full overview

The JPEG algorithm performs quantization of the DCT coefficients of non-overlapping 8×8 blocks of images, setting many of these coefficients to zero. The method takes advantage of these facts and identifies the presence of a JPEG grid when a significant number of DCT zeros are observed for a given grid origin. More specifically, each pixel votes for 1 of the 64 possible 8x8 grid positions, this being the one with more zero coefficients found, or voting for no grid in the case of a tie.

The algorithm later includes a statistical validation step according to the a-contrario theory of Desolneux, Moisan and Morel[1], which associates a number of false alarms (NFA) to each tampering detection. The detections are obtained by a threshold of the NFA.

## Usage

Add later usage of method 

## Results on benchmarking dataset

Add results of all metrics in our own benchmarking dataset

## Results on common datasets

Add results on common datasets 

## References:

[1] Agnès Desolneux, Lionel Moisan, and Jean-Michel Morel. From gestalt theory
to Image Analysis: A probabilistic approach. Springer, 2011.

## Citation

```tex
@inproceedings{zero,
  TITLE = {{JPEG Grid Detection based on the Number of DCT Zeros and its Application to Automatic and Localized Forgery Detection}},
  AUTHOR = {Nikoukhah, Tina and Anger, J and Ehret, T and Colom, Miguel and Morel, J M and Grompone von Gioi, R},
  URL = {https://hal.science/hal-03859737},
  BOOKTITLE = {{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops}},
  ADDRESS = {Long Beach, United States},
  YEAR = {2019},
  MONTH = Jun,
  PDF = {https://hal.science/hal-03859737/file/Nikoukhah_JPEG_Grid_Detection_based_on_the_Number_of_DCT_Zeros_CVPRW_2019_paper.pdf},
  HAL_ID = {hal-03859737},
  HAL_VERSION = {v1},
}
```