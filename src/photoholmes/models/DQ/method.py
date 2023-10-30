import numpy as np
from numpy.typing import NDArray

from photoholmes.models.base import BaseMethod
from photoholmes.models.DQ.utils import (
    ZIGZAG,
    fft_period,
    histogram_period,
    upsample_heatmap,
)


class DQ(BaseMethod):
    def __init__(self, number_frecs: int = 10, alpha: float = 1.0, **kwargs) -> None:
        """
        Initialize the DQ class.

        :param number_frecs: Number of frequencies, defaults to 10.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.number_frecs = number_frecs
        self.alpha = alpha

    def predict(self, dct_coefficients: NDArray) -> NDArray:
        """
        Predict the BPPM upsampled values.

        :param dct_coefficients: DCT coefficients.
        :return: BPPM upsampled values.
        """
        M, N = dct_coefficients.shape[1:]
        BPPM = np.zeros((M // 8, N // 8))
        for channel in range(dct_coefficients.shape[0]):
            BPPM += self._calculate_BPPM_channel(
                dct_coefficients[channel], ZIGZAG[: self.number_frecs]
            )
        BPPM_norm = BPPM / len(dct_coefficients)
        BPPM_upsampled = upsample_heatmap(BPPM_norm, (M, N))
        return BPPM_upsampled

    def _detect_period(self, histogram: np.ndarray) -> int:
        """
        Detect the period of the histogram.

        :param histogram: Input histogram.
        :return: Detected period.
        """
        p_H = histogram_period(histogram, self.alpha)
        p_fft = fft_period(histogram)
        p = min(p_H, p_fft)
        return p

    def _calculate_Pu(
        self, coefficients_f: np.ndarray, histogram: np.ndarray, period: int
    ) -> np.ndarray:
        """
        Calculate Pu values for a given frequency.

        :param coefficients_f: Coefficients for a given frequency.
        :param histogram: Input histogram for a given frequency.
        :param period: Detected period for a given frequency.
        :return: Calculated Pu values for a given frequency.
        """
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

    def _calculate_BPPM_f(self, DCT_coefficients_f: np.ndarray) -> np.ndarray:
        """
        Calculate BPPM values for given DCT coefficients for a given frequency..

        :param DCT_coefficients_f: DCT coefficients for a given frequency..
        :return: Calculated BPPM values for a given frequency..
        """
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

    def _calculate_BPPM_channel(
        self, DCT_coefs: np.ndarray, fs: np.ndarray
    ) -> np.ndarray:
        """
        Calculate BPPM values for a given channel.

        :param DCT_coefs: DCT coefficients for the channel.
        :param fs: Frequency values.
        :return: Calculated BPPM values for the channel.
        """
        M, N = DCT_coefs.shape
        BPPM = np.zeros((len(fs), M // 8, N // 8))
        for i in range(len(fs)):
            DCT_coefficients_f = DCT_coefs[fs[i][0] :: 8, fs[i][1] :: 8]
            BPPM[i] = self._calculate_BPPM_f(DCT_coefficients_f)

        return BPPM.sum(axis=0) / self.number_frecs
