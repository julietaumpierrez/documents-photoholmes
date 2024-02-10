# FOCAL: Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering

This is the implementation of the [FOCAL](https://arxiv.org/pdf/2308.09307.pdf) paper. The original implementation could be found [here](https://github.com/HighwayWu/FOCAL/tree/main).

## Description

FOCAL is based on a simple but very effective paradigm of contrastive learning and unsupervised clustering for the image forgery detection.
Specifically, FOCAL:
1) Utilizes pixel-level contrastive learning to supervise the high-level forensic feature extraction in an image-by-image manner.
2) Employs an on-the-fly unsupervised clustering algorithm (instead of a trained one) to cluster the learned features into forged/pristine categories, further suppressing the cross-image influence from training data.
3) Allows to further boost the detection performance via simple feature-level concatenation without the need of retraining.

## Full overview

The paper, "Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering," introduces a novel approach named FOCAL (FOrensic ContrAstive cLustering) for image forgery detection.

This method addresses the limitations of traditional pixel classification algorithms by considering the relative nature of forged versus pristine pixels within an image.

FOCAL employs pixel-level contrastive learning to enhance high-level forensic feature extraction and uses an on-the-fly unsupervised clustering algorithm to categorize these features into forged or pristine, improving detection performance without the need for retraining.

The framework of FOCAL is shown in the following figure:

![FOCAL](
  framework.jpg
)

Extensive testing across six public datasets shows significant performance improvements over state-of-the-art methods.

The paper highlights the importance of the relative definition of forgery and pristine conditions within images, offering a fresh perspective and setting a new benchmark for future research in image forgery detection.
## Usage

Add later usage of method 

## Results on benchmarking dataset

Add results of all metrics in our own benchmarking dataset

## Results on common datasets

Add results on common datasets 

## Citation

```tex
@article{focal,
  title={Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering},
  author={H. Wu and Y. Chen and J. Zhou},
  year={2023}
}
```
