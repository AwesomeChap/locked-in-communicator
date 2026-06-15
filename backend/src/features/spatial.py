"""Common Spatial Patterns feature extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


class CSPFeatureExtractor:
    """Binary Common Spatial Patterns (CSP).

    CSP learns spatial filters that maximize variance for one class while
    minimizing it for the other. Features are log-normalized variances of the
    projected epochs, producing compact spatial-spectral vectors for LDA/SVMs.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        feature_cfg = config["features"]
        n_components = int(feature_cfg["n_csp_components"])
        if n_components < 2:
            raise ValueError("CSP requires at least two components.")
        self._n_components = n_components
        self._regularization = float(feature_cfg["covariance_regularization"])
        self.filters_: NDArray[np.float64] | None = None
        self.classes_: NDArray[np.int_] | None = None

    @staticmethod
    def _epoch_covariances(epochs: NDArray[np.float64]) -> NDArray[np.float64]:
        demeaned = epochs - epochs.mean(axis=-1, keepdims=True)
        covariances = np.matmul(demeaned, np.swapaxes(demeaned, -1, -2))
        trace = np.trace(covariances, axis1=-2, axis2=-1)
        return covariances / np.maximum(trace[:, None, None], np.finfo(float).eps)

    def fit(
        self,
        epochs: NDArray[np.float64],
        labels: NDArray[np.int_],
    ) -> "CSPFeatureExtractor":
        """Fit CSP filters from epoched data ``(epochs, channels, samples)``."""

        x = np.asarray(epochs, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 3:
            raise ValueError("CSP expects epochs shaped (n_epochs, n_channels, n_samples).")

        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("CSP currently supports exactly two classes.")
        self.classes_ = classes.astype(int)

        covariances = self._epoch_covariances(x)
        cov_a = covariances[y == classes[0]].mean(axis=0)
        cov_b = covariances[y == classes[1]].mean(axis=0)
        composite = cov_a + cov_b
        composite += self._regularization * np.eye(composite.shape[0])

        eigenvalues, eigenvectors = eigh(cov_a, composite)
        order = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        filters = eigenvectors[:, order].T
        self.filters_ = filters[: self._n_components]
        return self

    def transform(self, epochs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project epochs into log-variance CSP feature vectors."""

        if self.filters_ is None:
            raise RuntimeError("CSPFeatureExtractor must be fitted before transform().")
        x = np.asarray(epochs, dtype=np.float64)
        if x.ndim == 2:
            x = x[None, :, :]
        if x.ndim != 3:
            raise ValueError("Expected epochs shaped (n_epochs, n_channels, n_samples).")

        projected = np.matmul(self.filters_[None, :, :], x)
        variances = projected.var(axis=-1)
        normalized = variances / np.maximum(
            variances.sum(axis=1, keepdims=True),
            np.finfo(float).eps,
        )
        return np.log(np.maximum(normalized, np.finfo(float).eps))

    def fit_transform(
        self,
        epochs: NDArray[np.float64],
        labels: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        return self.fit(epochs, labels).transform(epochs)
