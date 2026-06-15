"""Synthetic EEG-like stream for offline validation and dashboard demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config import get_n_channels, get_sampling_rate


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for epoched mock BCI data."""

    epochs: NDArray[np.float64]
    labels: NDArray[np.int_]


class MockBCIStream:
    """Generate class-conditioned mu/beta rhythms with spatial structure.

    This simulator is intentionally simple but physiologically inspired: the
    positive class has stronger 10 Hz mu rhythm over one lateralized pattern,
    while the negative class has stronger 20 Hz beta rhythm over the opposite
    pattern. Noise and a weak common-mode nuisance rhythm are added so the CSP
    stage must learn spatial variance differences rather than a trivial offset.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._fs = get_sampling_rate(config)
        self._n_channels = get_n_channels(config)
        self._chunk_size = int(config["ingestion"]["chunk_size"])
        self._class_samples = int(
            round(config["simulation"]["class_duration_s"] * self._fs)
        )
        self._rng = np.random.default_rng(int(config["simulation"]["random_state"]))
        self._target_label: int | None = None
        self._is_paused = False
        self._yes_label = int(config["simulation"]["yes_label"])
        self._no_label = int(config["simulation"]["no_label"])
        self.current_label = self._no_label

        channel_axis = np.linspace(-1.0, 1.0, self._n_channels)
        self._yes_pattern = self._normalize(np.exp(-((channel_axis + 0.45) ** 2) / 0.12))
        self._no_pattern = self._normalize(np.exp(-((channel_axis - 0.45) ** 2) / 0.12))
        self._common_pattern = self._normalize(np.ones(self._n_channels))

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @staticmethod
    def _normalize(pattern: NDArray[np.float64]) -> NDArray[np.float64]:
        norm = np.linalg.norm(pattern)
        return pattern / norm if norm > 0 else pattern

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def stop(self) -> None:
        self._is_paused = True

    def set_target(self, target: str | None) -> None:
        """Force a simulated intent label or return to automatic alternation."""

        if target is None:
            self._target_label = None
            return
        normalized = target.upper()
        if normalized == str(self._config["feedback"]["yes_label"]).upper():
            self._target_label = self._yes_label
        elif normalized == str(self._config["feedback"]["no_label"]).upper():
            self._target_label = self._no_label
        else:
            raise ValueError(f"Unsupported target intent: {target!r}")

    def _label_for_sample(self, sample_index: int) -> int:
        if self._target_label is not None:
            return self._target_label
        block = sample_index // max(self._class_samples, 1)
        return self._yes_label if block % 2 == 0 else self._no_label

    def _synthesize(
        self,
        label: int,
        start_sample: int,
        n_samples: int,
    ) -> NDArray[np.float64]:
        t = (np.arange(n_samples, dtype=np.float64) + start_sample) / self._fs
        sim_cfg = self._config["simulation"]
        yes_freq = float(sim_cfg["yes_frequency_hz"])
        no_freq = float(sim_cfg["no_frequency_hz"])

        phase = self._rng.uniform(0.0, 2.0 * np.pi)
        common_phase = self._rng.uniform(0.0, 2.0 * np.pi)
        common = float(sim_cfg["nuisance_std"]) * np.sin(
            2.0 * np.pi * 14.0 * t + common_phase
        )

        if label == self._yes_label:
            source = 2.0 * np.sin(2.0 * np.pi * yes_freq * t + phase)
            pattern = self._yes_pattern
        else:
            source = 2.0 * np.sin(2.0 * np.pi * no_freq * t + phase)
            pattern = self._no_pattern

        signal = pattern[:, None] * source[None, :]
        signal += self._common_pattern[:, None] * common[None, :]
        signal += self._rng.normal(
            0.0,
            float(sim_cfg["noise_std"]),
            size=(self._n_channels, n_samples),
        )
        return signal.astype(np.float64, copy=False)

    def pull_chunk_direct(self, sample_counter: int = 0) -> NDArray[np.float64]:
        """Return the next mock chunk as ``(n_channels, n_samples)``."""

        label = self._label_for_sample(sample_counter)
        self.current_label = label
        return self._synthesize(label, sample_counter, self._chunk_size)

    def generate_dataset(self, n_epochs: int) -> SyntheticDataset:
        """Create balanced epoched data shaped ``(n_epochs, channels, samples)``."""

        epoch_samples = int(round(self._config["epoching"]["epoch_length_s"] * self._fs))
        labels = np.tile([self._yes_label, self._no_label], int(np.ceil(n_epochs / 2)))[
            :n_epochs
        ]
        self._rng.shuffle(labels)
        epochs = np.stack(
            [
                self._synthesize(int(label), idx * epoch_samples, epoch_samples)
                for idx, label in enumerate(labels)
            ],
            axis=0,
        )
        return SyntheticDataset(epochs=epochs, labels=labels.astype(int))
