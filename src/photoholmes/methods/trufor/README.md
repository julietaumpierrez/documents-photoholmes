# TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization

This is the implementation of the metho presented in [[1]](https://arxiv.org/abs/2212.10957).
The code contained in this library was adapted from the [original implementation](https://github.com/grip-unina/TruFor).

## Description

The paper presents a novel approach to detect and localize image forgeries. The method
extracts both high-level and low-level features through a transformer-based architecture
that combines the RGB image and a learned noise-sensitive fingerprint. The latter one
is a re-train of the noiseprint method [[2]](https://ieeexplore.ieee.org/document/8713484),
dubbed noiseprint++. The forgeries are detected as deviations from the expected regular pattern
that characterizes a pristine image.

On top of a pixel-level localization map and a whole-image integrity score, the method outputs
a reliability map that highlights areas where the localization predictions may be error-prone,
reducing false-alarms.


## Full overview

TODO: add full overview

## Usage

## Results on benchmarking dataset

## Results on common datasets

## Citation
```tex
@article{Cozzolino2019_Noiseprint,
  title={Noiseprint: A CNN-Based Camera Model Fingerprint},
  author={D. Cozzolino and L. Verdoliva},
  journal={IEEE Transactions on Information Forensics and Security},
  doi={10.1109/TIFS.2019.2916364},
  pages={144-159},
  year={2020},
  volume={15}
}
```

## References

[1] F. Guillaro, D. Cozzolino, A. Sud, N. Dufour, and L. Verdoliva, “TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization,” Dec. 2022, [Online]. Available: http://arxiv.org/abs/2212.10957

[2] D. Cozzolino and L. Verdoliva, “Noiseprint: a CNN-based camera model fingerprint,” Aug. 2018, [Online]. Available: http://arxiv.org/abs/1808.08396

