import numpy as np

from photoholmes.models.base import BaseMethod
from photoholmes.models.DQ.utils import fft_period, histogram_period

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


class DQ(BaseMethod):
    def __init__(self, number_frecs=10, **kwargs):
        super().__init__(**kwargs)
        self.number_frecs = number_frecs

    def predict(self, dct_coefficients: np.ndarray) -> np.ndarray:
        M, N = dct_coefficients.shape[1:]
        BPPM = np.zeros((M // 8, N // 8))
        for channel in range(dct_coefficients.shape[0]):
            BPPM += self._calculate_BPPM_channel(
                dct_coefficients[channel], ZIGZAG[: self.number_frecs]
            )
        return BPPM / len(dct_coefficients)

    def _detect_period(self, histogram):
        p_H = histogram_period(histogram)
        p_fft = fft_period(histogram)
        p = min(p_H, p_fft)
        return p

    def _calculate_Pu(self, coefficients_f, histogram, period):
        coefficients_f -= np.min(coefficients_f)
        M, N = coefficients_f.shape

        histogram_padded = np.pad(histogram, (0, period))
        coefficient_indices = coefficients_f.ravel()
        histogram_range = histogram_padded[
            coefficient_indices[:, np.newaxis] + np.arange(period) - 1
        ]

        Pu_f = histogram_range[:, 1] / np.sum(histogram_range, axis=1)
        Pu_f = Pu_f.reshape((M, N))

        return Pu_f

    def _calculate_BPPM_f(self, DCT_coefficients_f):
        hmax = np.max(DCT_coefficients_f)
        hmin = np.min(DCT_coefficients_f)
        if hmax - hmin:
            hist, _ = np.histogram(
                DCT_coefficients_f, bins=hmax - hmin, range=(hmin, hmax)
            )
            p = self._detect_period(hist[1:-1])
            if p != 1:
                Pu = self._calculate_Pu(DCT_coefficients_f, hist, p)
                Pt = 1 / p
                BPPM_f = Pt / (Pu + Pt)
                saturated = (DCT_coefficients_f == DCT_coefficients_f.min()) | (
                    DCT_coefficients_f == DCT_coefficients_f.max()
                )
                BPPM_f[saturated] = 0

                return BPPM_f
        return np.zeros_like(DCT_coefficients_f)

    def _calculate_BPPM_channel(self, DCT_coefs, fs):
        M, N = DCT_coefs.shape
        BPPM = np.zeros((len(fs), M // 8, N // 8))
        for i in range(len(fs)):
            DCT_coefficients_f = DCT_coefs[fs[i][0] :: 8, fs[i][1] :: 8]
            BPPM[i] = self._calculate_BPPM_f(DCT_coefficients_f)

        return BPPM.sum(axis=0) / self.number_frecs
