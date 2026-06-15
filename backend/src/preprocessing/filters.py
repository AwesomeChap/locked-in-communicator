"""Vectorized EEG preprocessing filters."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from config import get_sampling_rate


class SignalPreprocessor:
    """Zero-phase bandpass and notch filtering for EEG epochs.

    The default 4th-order Butterworth bandpass isolates 8-30 Hz sensorimotor
    mu/beta rhythms. Filtering is forward-backward, so phase delay is canceled;
    this is appropriate for completed epochs and offline validation. For strict
    causal feedback, replace this stage with a stateful one-pass filter.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._fs = get_sampling_rate(config)
        dsp = config["preprocessing"]
        low = float(dsp["bandpass_low_hz"])
        high = float(dsp["bandpass_high_hz"])
        order = int(dsp["butterworth_order"])
        nyquist = self._fs / 2.0
        if not 0.0 < low < high < nyquist:
            raise ValueError(
                f"Invalid bandpass bounds ({low}, {high}) for fs={self._fs}."
            )

        self._bandpass_sos = butter(
            order,
            [low, high],
            btype="bandpass",
            fs=self._fs,
            output="sos",
        )
        notch_hz = float(config["signal"]["line_noise_hz"])
        q = float(dsp["notch_quality_factor"])
        self._notch_ba = (
            iirnotch(notch_hz, q, fs=self._fs)
            if 0.0 < notch_hz < nyquist
            else None
        )

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Filter data along the last axis.

        Parameters
        ----------
        data:
            Continuous data ``(channels, samples)`` or epoched data
            ``(epochs, channels, samples)``.
        """

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim not in (2, 3):
            raise ValueError("Expected data shaped (channels, samples) or epochs thereof.")

        filtered = sosfiltfilt(self._bandpass_sos, arr, axis=-1)
        if self._notch_ba is None:
            return filtered
        b, a = self._notch_ba
        return filtfilt(b, a, filtered, axis=-1)
