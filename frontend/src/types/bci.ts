/**
 * Shared TypeScript types for the BCI WebSocket protocol.
 *
 * The Python server broadcasts ``BCIMetricsMessage`` payloads every ~500 ms.
 * The frontend sends ``BCICommand`` objects back to control the simulation.
 */

export type ConnectionStatus = 'CONNECTED' | 'RECONNECTING' | 'DISCONNECTED';

export type SystemState = 'FITTING' | 'RUNNING' | 'PAUSED';

export type IntentClass = 'YES' | 'NO';

export type IntentTarget = 'YES' | 'NO' | 'AUTO';

/** Metrics payload broadcast by the Python server on each classified epoch. */
export interface BCIMetricsMessage {
  type: 'metrics';
  /** ISO-8601 UTC timestamp of the classification event. */
  timestamp: string;
  /** The class predicted by the sLDA classifier. */
  predicted_class: IntentClass;
  /** Probability of the winning class (0–1). */
  confidence: number;
  /** True when confidence >= configured threshold (default 0.70). */
  high_confidence: boolean;
  /** Simulator ground-truth class for this epoch. */
  ground_truth: IntentClass;
  /** Rolling classification accuracy over the last 50 epochs. */
  overall_accuracy: number;
  /** Total number of epochs classified since last RESET. */
  epoch_count: number;
  /** Current target intent forced by the user (or AUTO for alternating). */
  target_intent: IntentTarget;
  /** Current pipeline system state. */
  system_state: SystemState;
  /**
   * Downsampled raw EEG signal snapshot from channel C3.
   * Array length is ~120 samples — enough to animate a smooth waveform.
   */
  raw_signal_snapshot: number[];
}

/** Initial state snapshot sent when a client first connects. */
export interface BCIStateMessage {
  type: 'state';
  system_state: SystemState;
  target_intent: IntentTarget;
  epoch_count: number;
}

export type BCIServerMessage = BCIMetricsMessage | BCIStateMessage;

/** Commands the frontend can send to the Python server. */
export type BCICommand =
  | { command: 'START' }
  | { command: 'PAUSE' }
  | { command: 'SET_TARGET'; value: IntentTarget }
  | { command: 'RESET' };
