#!/usr/bin/env python3
"""Validate the BCI pipeline on a real acquired EDF recording.

This entry point parses the raw lab recording
``EEG Recoding 230620261205.edf``, reconstructs YES/NO motor-imagery trials
from the analog marker channel via peak detection, and runs the project's
core :class:`BCIPipeline` (CSP + shrinkage LDA) under Stratified 5-Fold
cross-validation.

The recording uses a 512 Hz, 17-signal montage. We only consume the
sensorimotor pair C3/C4 (linked-mastoid referenced against A1/A2) plus the
analog MARKER channel that encodes trial onsets as stepped pulses ("Hügels").

Marker amplitude tiers map to the project's existing class convention:

* low-amplitude pulse  -> Class 1 (YES / left-arm imagery)
* high-amplitude pulse -> Class 0 (NO  / right-arm imagery)

Run from ``backend/``::

    python3 scripts/process_real_data.py
    python3 scripts/process_real_data.py --edf /path/to/recording.edf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import mne

# --- Make the in-tree backend package importable (mirrors validate_pipeline.py) ---
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from config import load_config  # noqa: E402
from pipeline import BCIPipeline  # noqa: E402
from preprocessing import EpochEngine, SignalPreprocessor  # noqa: E402


# --- Acquisition layout discovered in the EDF header ---------------------------
# Channel indices are fixed by the recording hardware, not by our config.
CHANNEL_INDEX = {
    "C3": 2,
    "C4": 6,
    "A1": 10,   # left mastoid reference
    "A2": 14,   # right mastoid reference
    "MARKER": 16,  # analog peak / trial-onset channel
}

# Stable identifier shared with the frontend dataset dropdown.
DATASET_ID = "eeg_recording_230620261205"

EDF_FILENAME = "EEG Recoding 230620261205.edf"

# Points sent in the cleaned C3/C4 waveform snapshot for the live chart.
SNAPSHOT_POINTS = 120


def _edf_candidates() -> list[Path]:
    """Ordered locations to search for the recording.

    Resolution must work in three contexts: local dev (file sits next to the
    repo), the bundled Docker image (file copied into ``backend/data/`` and
    shipped inside the container), and any custom deployment via the
    ``REAL_EDF_PATH`` environment variable.
    """

    candidates: list[Path] = []
    env_path = os.environ.get("REAL_EDF_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            BACKEND_ROOT / "data" / EDF_FILENAME,  # bundled — works in Docker
            REPO_ROOT.parent / EDF_FILENAME,        # original dev location
            REPO_ROOT / EDF_FILENAME,
            Path.cwd() / EDF_FILENAME,
        ]
    )
    return candidates


def resolve_edf_path() -> Path:
    """Return the first existing recording location, or raise with guidance."""

    for candidate in _edf_candidates():
        expanded = candidate.expanduser()
        if expanded.is_file():
            return expanded.resolve()
    searched = "\n  ".join(str(c) for c in _edf_candidates())
    raise FileNotFoundError(
        f"Could not locate '{EDF_FILENAME}'. Set the REAL_EDF_PATH environment "
        f"variable or place the file in backend/data/. Searched:\n  {searched}"
    )

# Trial / experiment geometry.
EPOCH_LENGTH_S = 2.0          # 2-second imagery window per the protocol
MARKER_MIN_HEIGHT = 2.0       # ignore baseline (0) and tiny analog jitter
MARKER_MIN_DISTANCE_S = 2.0   # rest period: never two onsets inside one trial

# Class convention shared with config.py (simulation.yes_label / no_label).
YES_LABEL = 1  # low-amplitude marker pulse
NO_LABEL = 0   # high-amplitude marker pulse


def load_referenced_signals(
    edf_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Load the EDF and build linked-mastoid referenced C3/C4 plus the marker.

    Returns
    -------
    eeg:
        ``(2, n_samples)`` array of ``[C3 - A1, C4 - A2]`` in volts.
    marker:
        ``(n_samples,)`` raw analog marker channel.
    fs:
        Sampling rate in Hz, read from the file (do not assume the config value).
    """

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    data = raw.get_data()  # (n_channels, n_samples), vectorized read

    c3 = data[CHANNEL_INDEX["C3"]]
    c4 = data[CHANNEL_INDEX["C4"]]
    a1 = data[CHANNEL_INDEX["A1"]]
    a2 = data[CHANNEL_INDEX["A2"]]

    # Linked-mastoid (bipolar) re-reference removes common-mode noise.
    eeg = np.vstack([c3 - a1, c4 - a2]).astype(np.float64)
    marker = data[CHANNEL_INDEX["MARKER"]].astype(np.float64)
    return eeg, marker, fs


def detect_marker_trials(
    marker: NDArray[np.float64],
    fs: float,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Detect trial onsets on the analog marker channel and tier them by height.

    Uses ``scipy.signal.find_peaks`` with a minimum height (rejects baseline /
    artifacts) and a minimum inter-peak distance equal to the trial rest period
    (so a single stepped pulse is never counted as multiple trials).

    The two amplitude tiers are split at the midpoint of the observed peak
    heights: low pulses -> YES (Class 1), high pulses -> NO (Class 0).
    """

    distance = max(1, int(round(MARKER_MIN_DISTANCE_S * fs)))
    peaks, props = find_peaks(
        marker,
        height=MARKER_MIN_HEIGHT,
        distance=distance,
    )
    heights = props["peak_heights"]
    if peaks.size == 0:
        raise RuntimeError("No marker peaks detected; check the MARKER channel.")

    # Data-driven tier boundary: midway between the lowest and highest pulse.
    tier_threshold = (heights.min() + heights.max()) / 2.0
    # Degenerate case: a single amplitude level -> everything is one class.
    labels = np.where(heights < tier_threshold, YES_LABEL, NO_LABEL).astype(int)
    return peaks.astype(int), labels


def build_epochs(
    eeg: NDArray[np.float64],
    onsets: NDArray[np.int_],
    labels: NDArray[np.int_],
    fs: float,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Slice 2-second epochs at each marker onset (vectorized, no signal loops).

    Onsets whose epoch window would run past the recording are dropped, and the
    label vector is filtered in lockstep so epochs and labels stay aligned.
    """

    length = int(round(EPOCH_LENGTH_S * fs))
    valid = (onsets >= 0) & (onsets + length <= eeg.shape[1])
    onsets = onsets[valid]
    labels = labels[valid]

    config = _build_config(fs)
    epoch_engine = EpochEngine(config)
    epochs = epoch_engine.extract(eeg, onsets)
    return epochs, labels


def _build_config(fs: float) -> dict[str, Any]:
    """Adapt the project config to this recording (512 Hz, 2-channel montage)."""

    return load_config(
        {
            "signal": {
                "sampling_rate_hz": fs,
                "n_channels": 2,
                "channel_names": ["C3-A1", "C4-A2"],
                "line_noise_hz": 50.0,
            },
            # The pipeline's standard 8-30 Hz, 4th-order Butterworth bandpass.
            "preprocessing": {
                "bandpass_low_hz": 8.0,
                "bandpass_high_hz": 30.0,
                "butterworth_order": 4,
            },
            "epoching": {
                "epoch_length_s": EPOCH_LENGTH_S,
                "epoch_offset_s": 0.0,
            },
            # Only two channels are available, so CSP yields two components.
            "features": {"n_csp_components": 2},
        }
    )


def clean_signal_snapshot(
    epochs: NDArray[np.float64],
    config: dict[str, Any],
    n_points: int = SNAPSHOT_POINTS,
) -> tuple[list[float], list[float]]:
    """Return short, bandpass-cleaned C3/C4 traces for the live line chart.

    Filters the epochs with the pipeline's own 8-30 Hz Butterworth stage and
    returns a downsampled slice (in microvolts) of the most active epoch, so the
    UI shows genuine processed brainwaves rather than synthetic placeholders.
    """

    if epochs.shape[0] == 0:
        return [], []

    filtered = SignalPreprocessor(config).transform(np.asarray(epochs, np.float64))
    # Pick the liveliest epoch (max combined C3/C4 variance) for a clear trace.
    variance = filtered.var(axis=-1).sum(axis=1)
    pick = int(np.argmax(variance))
    trace = filtered[pick]  # (2, n_samples)

    step = max(1, trace.shape[1] // n_points)
    # Vectorized downsample + V -> microvolt scaling.
    decimated = (trace[:, ::step][:, :n_points] * 1e6).round(3)
    return decimated[0].tolist(), decimated[1].tolist()


def out_of_fold_predictions(
    config: dict[str, Any],
    epochs: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Collect out-of-fold predictions using the same CV splits as the pipeline.

    Mirrors :meth:`BCIPipeline.cross_validate` fold-for-fold so the resulting
    confusion matrix is consistent with the reported fold accuracies, while
    exposing the predictions needed for the frontend confusion matrix.
    """

    cv_cfg = config["cross_validation"]
    splitter = StratifiedKFold(
        n_splits=int(cv_cfg["n_splits"]),
        shuffle=bool(cv_cfg["shuffle"]),
        random_state=int(cv_cfg["random_state"]),
    )
    x = np.asarray(epochs, dtype=np.float64)
    y = np.asarray(labels, dtype=int)

    y_true = np.empty_like(y)
    y_pred = np.empty_like(y)
    cursor = 0
    for train_idx, test_idx in splitter.split(x, y):
        fold = BCIPipeline(config)
        fold.fit(x[train_idx], y[train_idx])
        filtered = fold._preprocessor.transform(x[test_idx])
        features = fold._feature_extractor.transform(filtered)
        predicted = fold._classifier.predict(features)
        n = test_idx.size
        y_true[cursor : cursor + n] = y[test_idx]
        y_pred[cursor : cursor + n] = predicted
        cursor += n

    return y_true, y_pred


def save_metrics(
    output_paths: list[Path],
    fold_accuracies: list[float],
    mean_acc: float,
    std_acc: float,
    matrix: list[list[int]],
    total_epochs: int,
    class_counts: dict[str, int],
    fs: float,
) -> None:
    """Persist metrics in the shape the OfflineAnalyzer component consumes."""

    payload = {
        "id": "real-eeg-session",
        "name": "Real EEG Session (230620261205)",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "samplingRateHz": fs,
        "channels": ["C3-A1", "C4-A2"],
        # Matches the frontend CVResult interface.
        "folds": [round(float(a), 4) for a in fold_accuracies],
        # matrix[actual][predicted]; row/col order = [YES(1), NO(0)].
        "matrix": matrix,
        "totalEpochs": int(total_epochs),
        "meanAccuracy": round(float(mean_acc), 4),
        "stdAccuracy": round(float(std_acc), 4),
        "classCounts": class_counts,
    }
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")


def analyze_recording(
    edf_path: Path | str | None = None,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """Run the full real-data validation and return a UI-ready result payload.

    This is the reusable core invoked both by the CLI ``main()`` and by the
    WebSocket server's ``ANALYZE_OFFLINE`` command. It is intentionally
    synchronous/CPU-bound; callers that must stay responsive (the server)
    should run it in a thread-pool executor.

    Returns a JSON-serializable dict containing the cross-validation accuracy,
    a TP/FP/TN/FN confusion matrix, and cleaned C3/C4 signal snapshots.
    """

    if edf_path is not None:
        edf_path = Path(edf_path).expanduser().resolve()
        if not edf_path.is_file():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")
    else:
        edf_path = resolve_edf_path()

    eeg, marker, fs = load_referenced_signals(edf_path)
    onsets, marker_labels = detect_marker_trials(marker, fs)
    epochs, labels = build_epochs(eeg, onsets, marker_labels, fs)

    config = _build_config(fs)
    n_splits = int(config["cross_validation"]["n_splits"])
    if epochs.shape[0] < n_splits:
        raise RuntimeError(
            f"Only {epochs.shape[0]} usable epochs; need at least {n_splits} for CV."
        )

    pipeline = BCIPipeline(config)
    result = pipeline.cross_validate(epochs, labels)

    # Confusion matrix from out-of-fold predictions (consistent with the CV).
    y_true, y_pred = out_of_fold_predictions(config, epochs, labels)
    matrix = confusion_matrix(y_true, y_pred, labels=[YES_LABEL, NO_LABEL])
    oof_accuracy = float(accuracy_score(y_true, y_pred))
    # YES is the positive class; matrix rows/cols are ordered [YES, NO].
    tp, fn = int(matrix[0, 0]), int(matrix[0, 1])
    fp, tn = int(matrix[1, 0]), int(matrix[1, 1])

    snapshot_c3, snapshot_c4 = clean_signal_snapshot(epochs, config)

    n_yes = int(np.sum(labels == YES_LABEL))
    n_no = int(np.sum(labels == NO_LABEL))

    payload: dict[str, Any] = {
        "dataset": DATASET_ID,
        "recording": edf_path.name,
        "sampling_rate": fs,
        "channels": ["C3-A1", "C4-A2"],
        "accuracy": round(result.mean_accuracy, 4),
        "std_accuracy": round(result.std_accuracy, 4),
        "oof_accuracy": round(oof_accuracy, 4),
        "fold_accuracies": [round(float(a), 4) for a in result.fold_accuracies],
        # Both the explicit TP/FP/TN/FN block and the [YES, NO] matrix form.
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "matrix": matrix.tolist(),
        "total_epochs": int(epochs.shape[0]),
        "class_counts": {"yes": n_yes, "no": n_no},
        "signal_snapshot": snapshot_c3,
        "signal_snapshot_c4": snapshot_c4,
    }

    if save:
        backend_metrics = BACKEND_ROOT / "results" / "real_data_metrics.json"
        frontend_metrics = (
            REPO_ROOT / "frontend" / "public" / "real_data_metrics.json"
        )
        save_metrics(
            output_paths=[backend_metrics, frontend_metrics],
            fold_accuracies=result.fold_accuracies,
            mean_acc=result.mean_accuracy,
            std_acc=result.std_accuracy,
            matrix=matrix.tolist(),
            total_epochs=epochs.shape[0],
            class_counts={"yes": n_yes, "no": n_no},
            fs=fs,
        )
        payload["metrics_paths"] = [str(backend_metrics), str(frontend_metrics)]

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edf",
        type=Path,
        default=None,
        help=(
            "Path to the EDF recording. Defaults to REAL_EDF_PATH, then "
            "backend/data/, then the repo's parent directory."
        ),
    )
    args = parser.parse_args()

    try:
        payload = analyze_recording(args.edf, save=True)
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    cm = payload["confusion_matrix"]
    fold_scores = ", ".join(f"{s:.3f}" for s in payload["fold_accuracies"])
    print("LockedIn Communicator — real EDF validation")
    print(f"Recording: {payload['recording']}")
    print(f"Sampling rate: {payload['sampling_rate']:.0f} Hz")
    print("Channels: C3-A1, C4-A2 (linked-mastoid re-reference)")
    print(
        f"Detected trials: {payload['total_epochs']} "
        f"(YES={payload['class_counts']['yes']}, NO={payload['class_counts']['no']})"
    )
    print(f"Fold accuracies: [{fold_scores}]")
    print(
        "Cross-validation accuracy: "
        f"{payload['accuracy']:.3f} +/- {payload['std_accuracy']:.3f}"
    )
    print(f"Out-of-fold accuracy: {payload['oof_accuracy']:.3f}")
    print(
        "Confusion matrix: "
        f"TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}"
    )
    for path in payload.get("metrics_paths", []):
        print(f"Metrics written to: {path}")


if __name__ == "__main__":
    main()
