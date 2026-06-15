"""Command-line validation for the simulated BCI pipeline."""

from __future__ import annotations

from config import load_config
from pipeline import BCIPipeline


def main() -> None:
    config = load_config()
    pipeline = BCIPipeline(config)

    n_epochs = 240
    epochs, labels = pipeline.generate_synthetic_dataset(config, n_epochs=n_epochs)
    result = pipeline.cross_validate(epochs, labels)
    pipeline.fit(epochs, labels)
    inference = pipeline.infer_epoch(epochs[0])

    fold_scores = ", ".join(f"{score:.3f}" for score in result.fold_accuracies)
    print("LockedIn Communicator BCI validation")
    print(f"Epochs: {epochs.shape[0]}")
    print(f"Channels: {epochs.shape[1]}")
    print(f"Samples per epoch: {epochs.shape[2]}")
    print(f"Fold accuracies: [{fold_scores}]")
    print(
        "Cross-validation accuracy: "
        f"{result.mean_accuracy:.3f} +/- {result.std_accuracy:.3f}"
    )
    print(
        "Example inference: "
        f"{inference.text} "
        f"(label={inference.label}, confidence={inference.confidence:.3f})"
    )


if __name__ == "__main__":
    main()
