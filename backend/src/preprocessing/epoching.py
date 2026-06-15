"""Epoch extraction from continuous EEG and event markers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from config import get_sampling_rate


class EpochEngine:
    """Slice continuous streams into fixed windows around marker samples."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._fs = get_sampling_rate(config)
        epoch_cfg = config["epoching"]
        self.epoch_length_samples = int(round(float(epoch_cfg["epoch_length_s"]) * self._fs))
        self.epoch_offset_samples = int(round(float(epoch_cfg["epoch_offset_s"]) * self._fs))
        if self.epoch_length_samples <= 0:
            raise ValueError("Epoch length must be positive.")

    def extract(
        self,
        continuous: NDArray[np.float64],
        marker_samples: Iterable[int],
    ) -> NDArray[np.float64]:
        """Return epochs shaped ``(n_events, channels, samples)``.

        Markers are sample indices in the continuous stream. ``epoch_offset_s``
        can be negative to include pre-stimulus baseline samples.
        """

        data = np.asarray(continuous, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Continuous data must be shaped (channels, samples).")

        starts = np.asarray(list(marker_samples), dtype=int) + self.epoch_offset_samples
        stops = starts + self.epoch_length_samples
        valid = (starts >= 0) & (stops <= data.shape[1])
        starts = starts[valid]
        if starts.size == 0:
            return np.empty((0, data.shape[0], self.epoch_length_samples), dtype=np.float64)

        sample_offsets = np.arange(self.epoch_length_samples, dtype=int)
        indices = starts[:, None] + sample_offsets[None, :]
        return np.take(data, indices, axis=1).transpose(1, 0, 2)

    def sliding_windows(
        self,
        continuous: NDArray[np.float64],
        stride_samples: int | None = None,
    ) -> NDArray[np.float64]:
        """Create uniformly spaced windows for marker-free validation."""

        data = np.asarray(continuous, dtype=np.float64)
        stride = stride_samples or self.epoch_length_samples
        if stride <= 0:
            raise ValueError("Stride must be positive.")
        if data.shape[-1] < self.epoch_length_samples:
            return np.empty((0, data.shape[0], self.epoch_length_samples), dtype=np.float64)

        starts = np.arange(
            0,
            data.shape[-1] - self.epoch_length_samples + 1,
            stride,
            dtype=int,
        )
        sample_offsets = np.arange(self.epoch_length_samples, dtype=int)
        indices = starts[:, None] + sample_offsets[None, :]
        return np.take(data, indices, axis=1).transpose(1, 0, 2)
