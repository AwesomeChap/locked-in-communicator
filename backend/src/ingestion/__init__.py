"""Streaming and mock acquisition backends."""

from ingestion.lsl import LSLInletClient, LSLStreamConfig
from ingestion.simulation import MockBCIStream

__all__ = ["LSLInletClient", "LSLStreamConfig", "MockBCIStream"]
