"""End-to-end YES/NO BCI decoding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import load_config
from features import CSPFeatureExtractor
from inference import (
    BCIInference,
    FeedbackDispatcher,
    ShrinkageLDAClassifier,
)
from ingestion import MockBCIStream
from preprocessing import EpochEngine, SignalPreprocessor


@dataclass(frozen=True)
class CrossValidationResult:
    """Summary of offline stratified cross-validation."""

    fold_accuracies: list[float]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_accuracies))

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.fold_accuracies))


class BCIPipeline:
    """Modular orchestration of ingestion, DSP, CSP features, and sLDA inference."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = load_config(config)
        self._preprocessor = SignalPreprocessor(self._config)
        self._epoch_engine = EpochEngine(self._config)
        self._feature_extractor = CSPFeatureExtractor(self._config)
        self._classifier = ShrinkageLDAClassifier(self._config)
        self._mock_streamer = MockBCIStream(self._config)
        self._dispatcher = FeedbackDispatcher()
        self.is_fitted = False

    def register_action_handler(self, handler: Callable[[BCIInference], None]) -> None:
        """Register a callback for high-confidence YES/NO decisions."""

        self._dispatcher.register(handler)

    def fit(
        self,
        epochs: NDArray[np.float64],
        labels: NDArray[np.int_],
    ) -> "BCIPipeline":
        """Fit preprocessing-dependent features and classifier.

        Epochs must be shaped ``(n_epochs, n_channels, n_samples)`` and labels
        must contain two integer classes. Filtering is fit-free; CSP and sLDA
        are learned only from the provided training folds/epochs.
        """

        y = np.asarray(labels, dtype=int)
        filtered = self._preprocessor.transform(np.asarray(epochs, dtype=np.float64))
        features = self._feature_extractor.fit_transform(filtered, y)
        self._classifier.fit(features, y)
        self.is_fitted = True
        return self

    def predict_epoch(self, epoch: NDArray[np.float64]) -> tuple[int, float]:
        """Classify one raw epoch shaped ``(n_channels, n_samples)``."""

        inference = self.infer_epoch(epoch)
        return inference.label, inference.confidence

    def infer_epoch(self, epoch: NDArray[np.float64]) -> BCIInference:
        """Classify one epoch and emit high-confidence feedback callbacks."""

        if not self.is_fitted:
            raise RuntimeError("BCIPipeline must be fitted before inference.")
        filtered = self._preprocessor.transform(np.asarray(epoch, dtype=np.float64))
        features = self._feature_extractor.transform(filtered)
        probabilities = self._classifier.predict_proba(features)[0]
        class_index = int(np.argmax(probabilities))
        label = int(self._classifier._model.classes_[class_index])
        confidence = float(probabilities[class_index])
        inference = BCIInference(
            label=label,
            text=self._label_to_text(label),
            confidence=confidence,
            high_confidence=confidence
            >= float(self._config["feedback"]["confidence_threshold"]),
        )
        self._dispatcher.emit(inference)
        return inference

    def _label_to_text(self, label: int) -> str:
        yes_label = int(self._config["simulation"]["yes_label"])
        return (
            str(self._config["feedback"]["yes_label"])
            if label == yes_label
            else str(self._config["feedback"]["no_label"])
        )

    def cross_validate(
        self,
        epochs: NDArray[np.float64],
        labels: NDArray[np.int_],
    ) -> CrossValidationResult:
        """Evaluate the full train-time pipeline with Stratified K-Fold CV."""

        cv_cfg = self._config["cross_validation"]
        splitter = StratifiedKFold(
            n_splits=int(cv_cfg["n_splits"]),
            shuffle=bool(cv_cfg["shuffle"]),
            random_state=int(cv_cfg["random_state"]),
        )
        x = np.asarray(epochs, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        accuracies: list[float] = []

        for train_idx, test_idx in splitter.split(x, y):
            fold_pipeline = BCIPipeline(self._config)
            fold_pipeline.fit(x[train_idx], y[train_idx])
            filtered = fold_pipeline._preprocessor.transform(x[test_idx])
            features = fold_pipeline._feature_extractor.transform(filtered)
            predicted = fold_pipeline._classifier.predict(features)
            accuracies.append(float(accuracy_score(y[test_idx], predicted)))

        return CrossValidationResult(fold_accuracies=accuracies)

    def generate_synthetic_dataset(
        self,
        config: dict[str, Any] | None = None,
        n_epochs: int = 200,
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Generate a balanced synthetic EEG dataset for validation."""

        streamer = MockBCIStream(load_config(config or self._config))
        dataset = streamer.generate_dataset(n_epochs=n_epochs)
        return dataset.epochs, dataset.labels
