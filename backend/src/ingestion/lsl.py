"""Asynchronous Lab Streaming Layer ingestion utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LSLStreamConfig:
    """LSL stream selection and chunking parameters."""

    stream_type: str = "EEG"
    stream_name: str | None = None
    chunk_size: int = 25
    resolve_timeout_s: float = 5.0


class LSLInletClient:
    """Non-blocking wrapper around a pylsl StreamInlet.

    The client yields chunks shaped ``(n_channels, n_samples)`` to match the
    downstream DSP modules. ``pylsl`` is imported lazily so offline validation
    and CI can run without LSL libraries installed.
    """

    def __init__(self, config: LSLStreamConfig) -> None:
        self._config = config
        self._inlet: object | None = None

    def connect(self) -> None:
        """Resolve and connect to the configured LSL stream."""

        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError as exc:
            raise RuntimeError(
                "pylsl is required for live LSL ingestion. Install the optional "
                "dependency with `pip install pylsl`."
            ) from exc

        prop = "name" if self._config.stream_name else "type"
        value = self._config.stream_name or self._config.stream_type
        streams = resolve_byprop(prop, value, timeout=self._config.resolve_timeout_s)
        if not streams:
            raise RuntimeError(f"No LSL stream found for {prop}={value!r}.")
        self._inlet = StreamInlet(streams[0])

    async def chunks(self) -> AsyncIterator[NDArray[np.float64]]:
        """Yield live chunks as ``(n_channels, n_samples)`` arrays."""

        if self._inlet is None:
            self.connect()

        assert self._inlet is not None
        while True:
            samples, _timestamps = await asyncio.to_thread(
                self._inlet.pull_chunk,
                max_samples=self._config.chunk_size,
            )
            if not samples:
                await asyncio.sleep(0.001)
                continue
            yield np.asarray(samples, dtype=np.float64).T
