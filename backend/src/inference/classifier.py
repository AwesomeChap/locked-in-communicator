"""Low-latency statistical classification and feedback triggers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass(frozen=True)
class BCIInference:
    """Single binary BCI decision emitted by the pipeline."""

    label: int
    text: str
    confidence: float
    high_confidence: bool


class ShrinkageLDAClassifier:
    """Shrinkage Linear Discriminant Analysis for compact CSP features."""

    def __init__(self, config: dict[str, Any]) -> None:
        shrinkage = config["classifier"].get("shrinkage", "auto")
        self._model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)

    def fit(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int_],
    ) -> "ShrinkageLDAClassifier":
        self._model.fit(features, labels)
        return self

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        return self._model.predict(features).astype(int)

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.predict_proba(features)


class FeedbackDispatcher:
    """Register actions that run when a high-confidence decision occurs."""

    def __init__(self) -> None:
        self._handlers: list[Callable[[BCIInference], None]] = []

    def register(self, handler: Callable[[BCIInference], None]) -> None:
        self._handlers.append(handler)

    def emit(self, inference: BCIInference) -> None:
        if not inference.high_confidence:
            return
        for handler in self._handlers:
            handler(inference)
