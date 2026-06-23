"""Real-time WebSocket + static-file server for the BCI verification dashboard.

Serves the compiled React frontend as static files **and** the WebSocket
endpoint (``/ws``) from a single aiohttp process.  This lets the whole app
run as one Render web service without CORS or two-service complexity.

Entry-point: ``lockedin-verification-server`` (see pyproject.toml).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Set

import aiohttp
import numpy as np
from aiohttp import web

from config import load_config
from pipeline.bci_pipeline import BCIPipeline

# The reusable real-EDF analysis core lives in scripts/process_real_data.py.
# Add that directory to the path so the server can import it directly, per the
# integration spec ("import and trigger the logic in process_real_data.py").
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logger = logging.getLogger(__name__)

_HISTORY_LEN = 50
_EPOCH_INTERVAL_S = 0.5   # 2 Hz classification rate
_SNAPSHOT_POINTS = 120    # raw-signal points sent per epoch
# Render (and most PaaS) inject PORT; fall back to 8765 for local dev.
_WS_HOST = os.environ.get("WS_HOST", "0.0.0.0")
_WS_PORT = int(os.environ.get("PORT", "8765"))
# Path to the compiled React app.  Resolved relative to CWD so it works both
# locally (CWD = repo root) and on Render (also runs from repo root).
_STATIC_DIR = os.path.abspath(
    os.environ.get("STATIC_DIR", "frontend/dist")
)


# ---------------------------------------------------------------------------
# Thin adapter: lets _handler speak the same API regardless of WS library
# ---------------------------------------------------------------------------


class _AioWSAdapter:
    """Wraps aiohttp WebSocketResponse to match the interface _handler expects."""

    def __init__(self, ws: web.WebSocketResponse, request: web.Request) -> None:
        self._ws = ws
        self.remote_address = request.remote

    async def send(self, message: str) -> None:
        await self._ws.send_str(message)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:  # type: ignore[override]
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                yield msg.data
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class BCIVerificationServer:
    """Async WebSocket server wiring the BCI pipeline to a React dashboard.

    Lifecycle
    ---------
    1.  ``serve()`` fits the pipeline (in a thread-pool so the event loop
        stays responsive) then starts the simulation loop.
    2.  The simulation loop generates synthetic EEG epochs, runs inference,
        and broadcasts a JSON metrics payload to every connected client.
    3.  Each client connection is handled by ``_handler()``, which pushes an
        initial state snapshot and then dispatches inbound control commands.

    Control commands (JSON sent from the frontend)
    -----------------------------------------------
    ``{"command": "START"}``
        Resume the simulation loop.
    ``{"command": "PAUSE"}``
        Pause epoch generation (server stays alive, waveform freezes).
    ``{"command": "RESET"}``
        Clear rolling accuracy history and epoch counter.
    ``{"command": "ANALYZE_OFFLINE", "dataset": "<id>"}``
        Run an offline validation on a real recording and return the metrics.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = load_config(config)
        self._pipeline = BCIPipeline(self._config)
        self._clients: Set[Any] = set()

        self._system_state: str = "FITTING"
        self._epoch_count: int = 0

        # Sliding window of correct/incorrect flags
        self._accuracy_window: deque[bool] = deque(maxlen=_HISTORY_LEN)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_accuracy(self) -> float:
        if not self._accuracy_window:
            return 0.0
        return sum(self._accuracy_window) / len(self._accuracy_window)

    def _fit_pipeline_sync(self) -> None:
        """Blocking: generate synthetic data and fit the pipeline."""
        logger.info("Generating synthetic training data and fitting pipeline…")
        t0 = time.perf_counter()
        epochs, labels = self._pipeline.generate_synthetic_dataset(n_epochs=240)
        self._pipeline.fit(epochs, labels)
        elapsed = time.perf_counter() - t0
        logger.info("Pipeline fitted in %.2f s.", elapsed)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    async def _simulation_loop(self) -> None:
        streamer = self._pipeline._mock_streamer
        cfg = self._config
        epoch_samples = int(
            round(cfg["epoching"]["epoch_length_s"] * cfg["signal"]["sampling_rate_hz"])
        )
        yes_label = int(cfg["simulation"]["yes_label"])
        yes_text = str(cfg["feedback"]["yes_label"])
        no_text = str(cfg["feedback"]["no_label"])
        sample_counter = 0

        while True:
            try:
                if self._system_state != "RUNNING":
                    await asyncio.sleep(0.1)
                    continue

                # Synthesize one epoch using the streamer's internal logic
                gt_label = streamer._label_for_sample(sample_counter)
                raw_epoch = streamer._synthesize(gt_label, sample_counter, epoch_samples)
                streamer.current_label = gt_label
                sample_counter += epoch_samples

                # Run the full pipeline: filter → CSP → sLDA
                inference = self._pipeline.infer_epoch(raw_epoch)
                self._epoch_count += 1

                # Determine ground-truth text and track accuracy
                ground_truth_text = yes_text if gt_label == yes_label else no_text
                self._accuracy_window.append(inference.text == ground_truth_text)

                # Build a downsampled signal snapshot from channel 0
                channel_0 = raw_epoch[0, :]
                step = max(1, len(channel_0) // _SNAPSHOT_POINTS)
                snapshot: list[float] = channel_0[::step].tolist()

                payload: dict[str, Any] = {
                    "type": "metrics",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "predicted_class": inference.text,
                    "confidence": round(float(inference.confidence), 4),
                    "high_confidence": bool(inference.high_confidence),
                    "ground_truth": ground_truth_text,
                    "overall_accuracy": round(self._rolling_accuracy(), 4),
                    "epoch_count": self._epoch_count,
                    "system_state": self._system_state,
                    "raw_signal_snapshot": snapshot,
                }
                await self._broadcast(json.dumps(payload))
                await asyncio.sleep(_EPOCH_INTERVAL_S)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unhandled error in simulation loop.")
                await asyncio.sleep(0.2)

    # ------------------------------------------------------------------
    # WebSocket plumbing
    # ------------------------------------------------------------------

    async def _broadcast(self, message: str) -> None:
        """Send *message* to every connected client; prune stale sockets."""
        if not self._clients:
            return
        snapshot = set(self._clients)
        results = await asyncio.gather(
            *[c.send(message) for c in snapshot],
            return_exceptions=True,
        )
        for client, result in zip(snapshot, results):
            if isinstance(result, Exception):
                self._clients.discard(client)

    async def _broadcast_state(self) -> None:
        """Push the current control state so the dashboard updates immediately."""
        await self._broadcast(
            json.dumps(
                {
                    "type": "state",
                    "system_state": self._system_state,
                    "epoch_count": self._epoch_count,
                    "overall_accuracy": round(self._rolling_accuracy(), 4),
                }
            )
        )

    async def _handler(self, websocket: Any) -> None:
        """Manage one client: deliver initial state, then handle commands."""
        self._clients.add(websocket)
        addr = getattr(websocket, "remote_address", "unknown")
        logger.info("Client connected: %s  (total: %d)", addr, len(self._clients))

        # Immediately push current system state so the UI can render
        await websocket.send(json.dumps({
            "type": "state",
            "system_state": self._system_state,
            "epoch_count": self._epoch_count,
            "overall_accuracy": round(self._rolling_accuracy(), 4),
        }))

        try:
            async for raw in websocket:
                await self._dispatch_command(raw, websocket)
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected: %s  (total: %d)", addr, len(self._clients))

    async def _dispatch_command(self, raw: str, websocket: Any) -> None:
        try:
            msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON message received: %r", raw[:120])
            return

        command = str(msg.get("command", "")).upper()
        handled = False
        if command == "START":
            self._system_state = "RUNNING"
            logger.info("Simulation started.")
            handled = True
        elif command == "PAUSE":
            self._system_state = "PAUSED"
            logger.info("Simulation paused.")
            handled = True
        elif command == "RESET":
            self._accuracy_window.clear()
            self._epoch_count = 0
            logger.info("Stats reset.")
            handled = True
        elif command == "ANALYZE_OFFLINE":
            # Offload to a background task so the analysis (EDF load + CSP/LDA
            # cross-validation) never blocks this client's receive loop or the
            # shared event loop.
            dataset = str(msg.get("dataset", ""))
            asyncio.create_task(self._run_offline_analysis(dataset, websocket))
        else:
            logger.warning("Unknown command: %r", command)

        if handled:
            await self._broadcast_state()

    async def _run_offline_analysis(self, dataset: str, websocket: Any) -> None:
        """Validate a real recording off the event loop and return the metrics.

        Currently only the lab EDF recording is wired in. The CPU/IO-bound
        analysis runs in the default thread-pool executor so the WebSocket
        server stays responsive; the structured result is then pushed back to
        the requesting client.
        """

        logger.info("Offline analysis requested for dataset %r.", dataset)
        try:
            from process_real_data import DATASET_ID, analyze_recording

            if dataset != DATASET_ID:
                raise ValueError(f"Unknown offline dataset: {dataset!r}")

            loop = asyncio.get_running_loop()
            # save=False: the live request does not need to rewrite the JSON
            # artifacts that the CLI script maintains.
            result = await loop.run_in_executor(
                None, lambda: analyze_recording(save=False)
            )
            await websocket.send(json.dumps({"type": "offline_result", **result}))
            logger.info(
                "Offline analysis complete for %r: acc=%.3f over %d epochs.",
                dataset,
                result["accuracy"],
                result["total_epochs"],
            )
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            logger.exception("Offline analysis failed for %r.", dataset)
            try:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "offline_error",
                            "dataset": dataset,
                            "message": str(exc),
                        }
                    )
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Entry-point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # aiohttp route handlers
    # ------------------------------------------------------------------

    async def _ws_route(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self._handler(_AioWSAdapter(ws, request))
        return ws

    async def _static_route(self, request: web.Request) -> web.Response:
        """Serve static files; fall back to index.html for SPA deep-links."""
        path = request.match_info.get("path", "")
        target = os.path.join(_STATIC_DIR, path) if path else _STATIC_DIR
        if os.path.isfile(target):
            return web.FileResponse(target)
        index = os.path.join(_STATIC_DIR, "index.html")
        if os.path.isfile(index):
            return web.FileResponse(index)
        return web.Response(status=404, text="Not found — did you build the frontend?")

    # ------------------------------------------------------------------
    # Entry-point
    # ------------------------------------------------------------------

    async def serve(self, host: str = _WS_HOST, port: int = _WS_PORT) -> None:
        """Fit the pipeline, then serve HTTP + WebSocket until cancelled."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._fit_pipeline_sync)
        self._system_state = "PAUSED"

        sim_task = asyncio.create_task(self._simulation_loop())

        app = web.Application()
        app.router.add_get("/ws", self._ws_route)
        app.router.add_get("/{path:.*}", self._static_route)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(
            "BCI server running at http://%s:%d  (WS → /ws, UI → /)",
            host, port,
        )
        try:
            await asyncio.Future()  # run forever
        except asyncio.CancelledError:
            pass
        finally:
            sim_task.cancel()
            try:
                await sim_task
            except asyncio.CancelledError:
                pass
            await runner.cleanup()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    server = BCIVerificationServer()
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
