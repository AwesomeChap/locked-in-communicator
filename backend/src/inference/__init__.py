"""Classification and inference utilities."""

from inference.classifier import (
    BCIInference,
    FeedbackDispatcher,
    ShrinkageLDAClassifier,
)

__all__ = ["BCIInference", "FeedbackDispatcher", "ShrinkageLDAClassifier"]
