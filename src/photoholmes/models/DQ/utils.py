import cv2
import numpy as np
from scipy.signal import find_peaks


def histogram_period(dct_histogram: np.ndarray, alpha: float = 1) -> int:
    """
    Calculate the period of a DCT histogram using a histogram-based method.

    :param dct_histogram: Input DCT histogram.
    :param alpha: Weighting factor, defaults to 1.
    :return: Detected period.
    """
    Hmax = 0
    period = 1
    k0 = np.argmax(dct_histogram)
    kmin, kmax = 0, len(dct_histogram) - 1

    for p in range(1, kmax // 20):
        imin, imax = np.floor((kmin - k0) / p), np.ceil((kmax - k0) / p)
        i = np.arange(imin, imax, dtype=int)
        H = np.sum(dct_histogram[i * p + k0] ** alpha) / (imax - imin + 1)
        if H > Hmax:
            Hmax = H
            period = p

    return period


def fft_period(dct_histogram: np.ndarray) -> int:
    """
    Calculate the period of a DCT histogram using FFT.

    :param dct_histogram: Input DCT histogram.
    :return: Detected period.
    """
    spectrogram = np.abs(np.fft.fftshift(np.fft.fft(dct_histogram)))
    log_spectrogram = np.log(spectrogram)
    c = len(log_spectrogram) // 2

    peaks = find_peaks(log_spectrogram[c - 1 :], distance=10)[0]
    main_peaks = peaks[log_spectrogram[peaks].argsort()[-1:-5:-1]]

    if len(main_peaks) > 1:
        main_peak = np.sort(main_peaks)[1]
        period = np.round(len(dct_histogram) / main_peak).astype(int)
    else:
        period = 1

    return period


def upsample_heatmap(heatmap: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Upsample a heatmap to match the given image shape.

    :param heatmap: Input heatmap.
    :param image_shape: Desired output shape.
    :return: Upsampled heatmap.
    """
    augmented_heatmap = cv2.resize(
        heatmap, (image_shape[1], image_shape[0]), interpolation=0
    )
    return augmented_heatmap


ZIGZAG = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
]
