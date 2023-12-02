# CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing

This is the implemenattion of the CAT-Net model presented [[1]](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html) and [[2]](https://arxiv.org/abs/2108.12947).
Both this paper introduce the same architechture, differing only in the trainig dataset used: v1 targeted only splicing while v2 targets splicing and copy-move.

The code contained in this library was derived from [the original implementation](https://github.com/mjkwon2021/CAT-Net), making only minor changes to fit our project structure.

## Description

CAT-Net is an end-to-end fully convolutional neural network designed to detect compression artifacts in images. CAT-Net combines both RGB and DCT streams, allowing it to simultaneously learn forensic features related to compression artifacts in these domains. Each stream considers multiple resolutions to deal with the various shapes and sizes of the spliced objects.


## Full overview

The RGB stream processes the color information of the image, which is often altered during image splicing, while the DCT stream analyzes the compression artifacts, which are usually introduced when an image is saved in a compressed format like JPEG. By analyzing both color information and compression artifacts, CAT-Net can better discern inconsistencies associated with splicing, compared to using only one of the streams.

Multiple resolution analysis refers to the processing of image data at various scales or resolutions. This is crucial in image splicing detection because spliced objects can come in different sizes and shapes. By analyzing the image at multiple resolutions, CAT-Net can adapt to various scales of splicing, making it more versatile and accurate in detecting spliced regions, irrespective of the size of the spliced objects.

# TODO: finish README
## Usage

Add later usage of method 

## Results on benchmarking dataset

Add results of all metrics in our own benchmarking dataset

## Results on common datasets

Add results on common datasets 

## Citation

* [1] CAT-Net v1 (WACV2021)
```tex
@inproceedings{kwon2021cat,
  title={CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing},
  author={Kwon, Myung-Joon and Yu, In-Jae and Nam, Seung-Hun and Lee, Heung-Kyu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={375--384},
  year={2021}
}
```

* [2] CAT-Net v2 (IJCV)
```tex
@article{kwon2022learning,
  title={Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization},
  author={Kwon, Myung-Joon and Nam, Seung-Hun and Yu, In-Jae and Lee, Heung-Kyu and Kim, Changick},
  journal={International Journal of Computer Vision},
  volume = {130},
  number = {8},
  pages={1875--1895},
  month = aug,
  year={2022},
  publisher={Springer},
  doi = {10.1007/s11263-022-01617-5}
}
```
