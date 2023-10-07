import numpy as np
from scipy.signal import find_peaks


def histogram_period(dct_histogram, alpha=1):
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


def fft_period(dct_histogram):
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
