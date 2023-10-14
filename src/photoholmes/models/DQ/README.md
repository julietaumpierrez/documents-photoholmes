# Fast, Automatic, and Fine-Grained Tampered JPEG Image Detection via DCT Coefficient Analysis

This repository contains the implementation of the method proposed in the paper "Fast, automatic and fine-grained tampered JPEG image detection via DCT coefficient analysis". The original paper can be found [here][http://mmlab.ie.cuhk.edu.hk/archive/2009/pr09_fast_automatic.pdf].
## Description

With the rapid advancement in image/video editing techniques, distinguishing tampered images from real ones has become a challenge. This method focuses on JPEG images and detects tampered regions by examining the double quantization effect hidden among the discrete cosine transform (DCT) coefficients.

The method offers:

    Automatic Location: It can automatically locate the tampered region without user intervention.
    Fine-Grained Detection: The detection is at the scale of 8x8 DCT blocks.
    Versatility: Capable of dealing with images tampered using various methods such as inpainting, alpha matting, texture synthesis, and other editing skills.
    Efficiency: Directly analyzes the DCT coefficients without fully decompressing the JPEG image, ensuring fast performance.

## Full Overview

The method is based on the DQ effect in forged JPEG images and can produce fine-grained output of the forgery region at the scale of 8x8 image blocks. The algorithm directly analyzes the DCT coefficients without fully decompressing the JPEG image, saving memory and computational load. The method is faster than bi-coherence based approaches and CRF based algorithms.
Usage

To use the method, follow the steps below:

    Load the JPEG image.
    Extract DCT coefficients and quantization matrices for YUV channels.
    Build histograms for each channel and each frequency.
    Compute the probability of each block being tampered based on the histograms.
    Generate the block posterior probability map (BPPM).
    Threshold the BPPM to identify tampered regions.
    Use a trained SVM to decide if the image is tampered.

## Results on Benchmarking Dataset

Add results of all metrics in benchmarking dataset here.
## Results on Common Datasets

Add results on common datasets here.
## Citation


```tex
@ARTICLE{FastJPEGDetection2009,
  author={Zhouchen Lin, Junfeng He, Xiaoou Tang, Chi-Keung Tang},
  journal={Pattern Recognition},
  title={Fast, automatic and fine-grained tampered JPEG image detection via DCT coefficient analysis},
  year={2009},
  doi={10.1016/j.patcog.2009.03.019}
}
```

## Implementation

The implementation for this method is provided in the DQ class, which inherits from the BaseMethod class. The method utilizes DCT coefficients to predict tampered regions in JPEG images. The core functionality revolves around the detection of the double quantization effect in the DCT coefficients.