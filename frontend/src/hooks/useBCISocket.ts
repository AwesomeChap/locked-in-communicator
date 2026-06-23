/**
 * useBCISocket — Real-time WebSocket hook for the BCI verification dashboard.
 *
 * Design decisions
 * ----------------
 * - **RAF throttling**: incoming messages are never applied directly in the
 *   `onmessage` handler. Instead the latest payload is stored in a ref and a
 *   single `requestAnimationFrame` is scheduled to flush it into React state.
 *   This caps UI updates at 60 fps regardless of server broadcast rate.
 *
 * - **Graceful reconnection**: on `close` the hook exponentially backs off
 *   from 1 s up to 15 s before retrying. On mount, the first attempt is
 *   immediate.
 *
 * - **Waveform rolling buffer**: incoming `raw_signal_snapshot` arrays are
 *   appended to a fixed-length ring buffer (`WAVEFORM_BUFFER_SIZE` samples).
 *   The chart reads this directly without any extra transformation.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  BCICommand,
  BCIMetricsMessage,
  BCIOfflineErrorMessage,
  BCIOfflineResultMessage,
  BCIServerMessage,
  BCIStateMessage,
  ConnectionStatus,
  IntentTarget,
  SystemState,
} from '../types/bci';

// Same-origin WebSocket: automatically uses wss:// in production (HTTPS) and
// ws:// in development.  In dev, Vite proxies /ws → ws://localhost:8765.
const DEFAULT_URL =
  (typeof window !== 'undefined'
    ? (window.location.protocol === 'https:' ? 'wss' : 'ws') +
      '://' + window.location.host + '/ws'
    : 'ws://localhost:8765');
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 15_000;
const WAVEFORM_BUFFER_SIZE = 300;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface BCISocketState {
  /** Live WebSocket connection status. */
  status: ConnectionStatus;
  /** Most recent metrics payload from the server (null before first epoch). */
  latest: BCIMetricsMessage | null;
  /** Rolling buffer of raw EEG samples for the waveform chart. */
  waveformBuffer: number[];
  /** Current system state as reported by the server (or locally assumed). */
  systemState: SystemState;
  /** Current target intent as reported by the server. */
  targetIntent: IntentTarget;
  /** Latest offline validation result (null until an ANALYZE_OFFLINE completes). */
  offlineResult: BCIOfflineResultMessage | null;
  /** Latest offline validation error, if the last request failed. */
  offlineError: BCIOfflineErrorMessage | null;
  /** Send a control command to the Python server. */
  sendCommand: (cmd: BCICommand) => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useBCISocket(url: string = DEFAULT_URL): BCISocketState {
  const [status, setStatus] = useState<ConnectionStatus>('DISCONNECTED');
  const [latest, setLatest] = useState<BCIMetricsMessage | null>(null);
  const [waveformBuffer, setWaveformBuffer] = useState<number[]>([]);
  const [systemState, setSystemState] = useState<SystemState>('FITTING');
  const [targetIntent, setTargetIntent] = useState<IntentTarget>('AUTO');
  const [offlineResult, setOfflineResult] =
    useState<BCIOfflineResultMessage | null>(null);
  const [offlineError, setOfflineError] =
    useState<BCIOfflineErrorMessage | null>(null);

  // Mutable refs — never cause re-renders
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(RECONNECT_BASE_MS);
  const rafRef = useRef<number | null>(null);
  const pendingRef = useRef<BCIMetricsMessage | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    function connect(): void {
      if (!mountedRef.current) return;
      setStatus('RECONNECTING');

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setStatus('CONNECTED');
        reconnectDelayRef.current = RECONNECT_BASE_MS;
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        wsRef.current = null;
        setStatus('DISCONNECTED');
        reconnectTimerRef.current = setTimeout(() => {
          reconnectDelayRef.current = Math.min(
            reconnectDelayRef.current * 2,
            RECONNECT_MAX_MS,
          );
          connect();
        }, reconnectDelayRef.current);
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onmessage = ({ data }: MessageEvent<string>) => {
        let msg: BCIServerMessage;
        try {
          msg = JSON.parse(data) as BCIServerMessage;
        } catch {
          return;
        }

        if (msg.type === 'state') {
          const state = msg as BCIStateMessage;
          setSystemState(state.system_state);
          setTargetIntent(state.target_intent);
          return;
        }

        // Offline analysis responses are one-off — apply them immediately
        // (no RAF throttling needed) and clear any prior error/result.
        if (msg.type === 'offline_result') {
          setOfflineError(null);
          setOfflineResult(msg as BCIOfflineResultMessage);
          return;
        }
        if (msg.type === 'offline_error') {
          setOfflineResult(null);
          setOfflineError(msg as BCIOfflineErrorMessage);
          return;
        }

        if (msg.type !== 'metrics') return;

        // Always store the latest payload — the RAF below decides when to render
        pendingRef.current = msg as BCIMetricsMessage;

        if (rafRef.current === null) {
          rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            const m = pendingRef.current;
            if (!m || !mountedRef.current) return;

            setLatest(m);
            setSystemState(m.system_state);
            setTargetIntent(m.target_intent);

            // Append new samples to the rolling waveform buffer
            setWaveformBuffer(prev => {
              const merged = prev.concat(m.raw_signal_snapshot);
              return merged.length > WAVEFORM_BUFFER_SIZE
                ? merged.slice(merged.length - WAVEFORM_BUFFER_SIZE)
                : merged;
            });
          });
        }
      };
    }

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
      wsRef.current?.close();
    };
  }, [url]);

  const sendCommand = useCallback((cmd: BCICommand) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
      return;
    }
    console.warn('[BCI] Cannot send command — WebSocket is not connected.', cmd);
  }, []);

  return {
    status,
    latest,
    waveformBuffer,
    systemState,
    targetIntent,
    offlineResult,
    offlineError,
    sendCommand,
  };
}
