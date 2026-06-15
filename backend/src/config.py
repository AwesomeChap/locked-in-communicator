"""Central configuration for hardware-agnostic BCI decoding.

All signal dimensions, sampling assumptions, filter parameters, and labels live
here so acquisition hardware can be swapped without touching pipeline code.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "signal": {
        "sampling_rate_hz": 250.0,
        "n_channels": 8,
        "channel_names": ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "Pz"],
        "line_noise_hz": 50.0,
    },
    "ingestion": {
        "stream_name": None,
        "stream_type": "EEG",
        "marker_stream_type": "Markers",
        "chunk_size": 25,
        "resolve_timeout_s": 5.0,
    },
    "preprocessing": {
        "bandpass_low_hz": 8.0,
        "bandpass_high_hz": 30.0,
        "butterworth_order": 4,
        "notch_quality_factor": 30.0,
    },
    "epoching": {
        "epoch_length_s": 2.0,
        "epoch_offset_s": 0.0,
    },
    "features": {
        "method": "csp",
        "n_csp_components": 4,
        "covariance_regularization": 1e-6,
    },
    "classifier": {
        "kind": "shrinkage_lda",
        "shrinkage": "auto",
    },
    "cross_validation": {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
    "simulation": {
        "random_state": 7,
        "class_duration_s": 2.0,
        "noise_std": 0.65,
        "nuisance_std": 0.15,
        "yes_frequency_hz": 10.0,
        "no_frequency_hz": 20.0,
        "yes_label": 1,
        "no_label": 0,
    },
    "feedback": {
        "yes_label": "YES",
        "no_label": "NO",
        "confidence_threshold": 0.7,
    },
}


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a mutable configuration dictionary.

    Parameters
    ----------
    overrides:
        Optional nested dictionary merged into :data:`DEFAULT_CONFIG`.
    """

    config = deepcopy(DEFAULT_CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def get_sampling_rate(config: dict[str, Any]) -> float:
    """Sampling rate in Hz used by filters, epoching, and simulators."""

    return float(config["signal"]["sampling_rate_hz"])


def get_n_channels(config: dict[str, Any]) -> int:
    """Configured EEG channel count."""

    return int(config["signal"]["n_channels"])
