/**
 * Dashboard — Main layout for the BCI Verification dashboard.
 *
 * Component tree
 * --------------
 * Dashboard
 *   ├─ ConnectionBadge
 *   ├─ IntentDisplay
 *   ├─ ConfidenceGauge (SVG radial arc)
 *   ├─ WaveformChart   (Recharts rolling line)
 *   ├─ StatsRow        (accuracy · epoch count · ground truth)
 *   └─ ControlPanel    (Start/Pause · target intent toggle)
 */

import { useEffect, useRef, useState } from 'react';
import {
  LineChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts';
import { useBCISocket } from '../hooks/useBCISocket';
import type { ConnectionStatus, IntentClass, IntentTarget, SystemState } from '../types/bci';
import { cn } from '../lib/utils';

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionBadge
// ─────────────────────────────────────────────────────────────────────────────

function ConnectionBadge({ status }: { status: ConnectionStatus }) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold tracking-widest uppercase',
        status === 'CONNECTED'
          && 'border-green-800 bg-green-950/70 text-green-400',
        status === 'RECONNECTING'
          && 'border-yellow-800 bg-yellow-950/70 text-yellow-400',
        status === 'DISCONNECTED'
          && 'border-red-900 bg-red-950/70 text-red-500',
      )}
    >
      <span
        className={cn(
          'inline-block h-2 w-2 rounded-full',
          status === 'CONNECTED'    && 'animate-pulse bg-green-400',
          status === 'RECONNECTING' && 'animate-pulse bg-yellow-400',
          status === 'DISCONNECTED' && 'bg-red-500',
        )}
      />
      {status}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// IntentDisplay
// ─────────────────────────────────────────────────────────────────────────────

interface IntentDisplayProps {
  predictedClass: IntentClass | null;
  highConfidence: boolean;
  /** Incremented on each high-confidence event to retrigger the CSS animation. */
  flashKey: number;
}

function IntentDisplay({ predictedClass, highConfidence, flashKey }: IntentDisplayProps) {
  const isYes    = predictedClass === 'YES';
  const hasClass = predictedClass !== null;

  return (
    <div className="flex flex-col items-center gap-3">
      <span className="text-xs font-semibold tracking-widest text-slate-500 uppercase">
        Decoded Intent
      </span>

      <div
        key={flashKey}
        className={cn(
          'flex h-32 w-48 items-center justify-center rounded-2xl border-2 transition-colors duration-300',
          !hasClass
            && 'border-slate-700 bg-slate-800/40',
          hasClass && isYes && highConfidence
            && 'intent-yes-flash border-green-500 bg-green-950/60',
          hasClass && isYes && !highConfidence
            && 'border-green-800/60 bg-green-950/30',
          hasClass && !isYes && highConfidence
            && 'intent-no-flash border-red-500 bg-red-950/60',
          hasClass && !isYes && !highConfidence
            && 'border-red-900/60 bg-red-950/30',
        )}
      >
        <span
          className={cn(
            'text-6xl font-black leading-none tracking-tight',
            !hasClass              && 'text-slate-600',
            hasClass && isYes      && 'text-green-400',
            hasClass && !isYes     && 'text-red-400',
          )}
        >
          {predictedClass ?? '—'}
        </span>
      </div>

      <div className="h-5">
        {highConfidence && hasClass && (
          <span
            className={cn(
              'text-xs font-semibold tracking-wider animate-fade-in',
              isYes ? 'text-green-500' : 'text-red-500',
            )}
          >
            ● HIGH CONFIDENCE
          </span>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfidenceGauge — SVG half-arc radial gauge
// ─────────────────────────────────────────────────────────────────────────────

function ConfidenceGauge({ confidence }: { confidence: number }) {
  // Half-circle arc: center (85,88), radius 68
  // M 17 88  A 68 68 0 0 1 153 88
  const R = 68;
  const ARC_LEN = Math.PI * R; // ≈ 213.6

  const clamped = Math.max(0, Math.min(1, confidence));
  const dashOffset = ARC_LEN * (1 - clamped);
  const pct = Math.round(clamped * 100);

  const strokeColor =
    clamped >= 0.7 ? '#22c55e'
    : clamped >= 0.5 ? '#f59e0b'
    : '#ef4444';

  return (
    <div className="flex flex-col items-center gap-3">
      <span className="text-xs font-semibold tracking-widest text-slate-500 uppercase">
        Confidence
      </span>

      <svg viewBox="0 0 170 100" className="w-44" aria-label={`Confidence: ${pct}%`}>
        {/* Track arc */}
        <path
          d="M 17 88 A 68 68 0 0 1 153 88"
          stroke="#1e293b"
          strokeWidth="14"
          fill="none"
          strokeLinecap="round"
        />
        {/* Progress arc */}
        <path
          d="M 17 88 A 68 68 0 0 1 153 88"
          stroke={strokeColor}
          strokeWidth="14"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={`${ARC_LEN} ${ARC_LEN}`}
          strokeDashoffset={dashOffset}
          style={{
            transition: 'stroke-dashoffset 0.45s cubic-bezier(0.4,0,0.2,1), stroke 0.3s ease',
          }}
        />
        {/* Percentage label */}
        <text
          x="85"
          y="78"
          textAnchor="middle"
          fill="#f1f5f9"
          fontSize="24"
          fontWeight="700"
          fontFamily="'JetBrains Mono', monospace"
        >
          {pct}%
        </text>
        <text
          x="85"
          y="95"
          textAnchor="middle"
          fill="#475569"
          fontSize="9"
          fontFamily="'Inter', sans-serif"
          letterSpacing="2"
        >
          CLASS PROBABILITY
        </text>
      </svg>

      {/* Linear backup bar */}
      <div className="w-44 overflow-hidden rounded-full bg-slate-800 h-1.5">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: strokeColor }}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// WaveformChart — rolling Recharts line for channel C3
// ─────────────────────────────────────────────────────────────────────────────

function WaveformChart({ buffer }: { buffer: number[] }) {
  const data = buffer.map((v, i) => ({ t: i, v }));

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold tracking-widest text-slate-500 uppercase">
          EEG Signal — Ch. C3
        </span>
        <span className="font-mono text-[10px] text-slate-600">250 Hz · C3</span>
      </div>

      <div className="h-28">
        {data.length > 2 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 2, bottom: 2, left: 0, right: 0 }}>
              <XAxis dataKey="t" hide />
              <YAxis domain={['auto', 'auto']} hide />
              <Line
                type="monotone"
                dataKey="v"
                stroke="#3b82f6"
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="flex items-center gap-2 text-sm text-slate-600">
              <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-slate-600 border-t-blue-500" />
              Awaiting signal stream…
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// StatsRow — rolling accuracy · epoch count · ground truth
// ─────────────────────────────────────────────────────────────────────────────

interface StatsRowProps {
  accuracy: number;
  epochCount: number;
  targetIntent: IntentTarget;
  groundTruth: IntentClass | null;
}

function StatsRow({ accuracy, epochCount, targetIntent, groundTruth }: StatsRowProps) {
  const accColor =
    accuracy >= 0.75 ? 'text-green-400'
    : accuracy >= 0.55 ? 'text-yellow-400'
    : 'text-red-400';

  const gtColor =
    groundTruth === 'YES' ? 'text-green-400'
    : groundTruth === 'NO' ? 'text-red-400'
    : 'text-slate-400';

  return (
    <div className="grid grid-cols-3 gap-3">
      {/* Rolling accuracy */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">
        <div className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
          Rolling Accuracy
        </div>
        <div className={cn('mt-1 font-mono text-3xl font-bold tabular-nums', accColor)}>
          {(accuracy * 100).toFixed(0)}%
        </div>
        <div className="mt-0.5 text-[10px] text-slate-600">last 50 epochs vs. ground truth</div>
      </div>

      {/* Epoch counter */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">
        <div className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
          Epochs Classified
        </div>
        <div className="mt-1 font-mono text-3xl font-bold tabular-nums text-blue-400">
          {epochCount.toLocaleString()}
        </div>
        <div className="mt-0.5 text-[10px] text-slate-600">since last reset</div>
      </div>

      {/* Ground truth / target */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">
        <div className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
          Ground Truth
        </div>
        <div className={cn('mt-1 font-mono text-3xl font-bold', gtColor)}>
          {groundTruth ?? '—'}
        </div>
        <div className="mt-0.5 text-[10px] text-slate-600">
          target: <span className="font-semibold text-slate-400">{targetIntent}</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ControlPanel
// ─────────────────────────────────────────────────────────────────────────────

interface ControlPanelProps {
  systemState: SystemState;
  targetIntent: IntentTarget;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onSetTarget: (t: IntentTarget) => void;
}

function ControlPanel({
  systemState,
  targetIntent,
  onStart,
  onPause,
  onReset,
  onSetTarget,
}: ControlPanelProps) {
  const isRunning = systemState === 'RUNNING';
  const isFitting = systemState === 'FITTING';

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-semibold tracking-widest text-slate-500 uppercase">
          Control Panel
        </span>
        <span
          className={cn(
            'rounded-full px-2.5 py-0.5 font-mono text-[10px] font-semibold tracking-wider uppercase',
            isRunning && 'bg-green-950 text-green-400',
            systemState === 'PAUSED'  && 'bg-yellow-950 text-yellow-400',
            isFitting                 && 'bg-blue-950 text-blue-400',
          )}
        >
          {systemState}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {/* Start / Pause button */}
        <button
          onClick={isRunning ? onPause : onStart}
          disabled={isFitting}
          className={cn(
            'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold',
            'transition-all duration-150 focus-visible:outline-none focus-visible:ring-2',
            isFitting && 'cursor-not-allowed opacity-50 bg-slate-700 text-slate-500',
            !isFitting && isRunning
              && 'bg-yellow-600 text-white hover:bg-yellow-500 active:bg-yellow-700',
            !isFitting && !isRunning
              && 'bg-blue-600 text-white hover:bg-blue-500 active:bg-blue-700',
          )}
        >
          {isFitting ? (
            <>
              <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />
              Fitting model…
            </>
          ) : isRunning ? (
            <>⏸ Pause Stream</>
          ) : (
            <>▶ Start Simulation</>
          )}
        </button>

        {/* Reset button */}
        <button
          onClick={onReset}
          className="rounded-lg bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-400
                     transition-all hover:bg-slate-700 hover:text-slate-200 active:bg-slate-900"
        >
          ↺ Reset Stats
        </button>

        {/* Divider */}
        <div className="h-8 w-px bg-slate-700" />

        {/* Target intent toggle */}
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-xs font-medium text-slate-500">Target:</span>
          {(['AUTO', 'YES', 'NO'] as IntentTarget[]).map((t) => (
            <button
              key={t}
              onClick={() => onSetTarget(t)}
              className={cn(
                'rounded-md px-3 py-1.5 text-xs font-semibold transition-all duration-150',
                targetIntent !== t
                  && 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200',
                targetIntent === t && t === 'AUTO'
                  && 'bg-blue-700/80 text-blue-100 ring-1 ring-blue-600',
                targetIntent === t && t === 'YES'
                  && 'bg-green-700/80 text-green-100 ring-1 ring-green-600',
                targetIntent === t && t === 'NO'
                  && 'bg-red-700/80 text-red-100 ring-1 ring-red-600',
              )}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <p className="mt-3 text-[10px] text-slate-600">
        Toggle <strong className="text-slate-500">YES</strong> or{' '}
        <strong className="text-slate-500">NO</strong> to force the simulator to stream a fixed
        intent class. <strong className="text-slate-500">AUTO</strong> alternates every 2 s.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard — root composition
// ─────────────────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const {
    status,
    latest,
    waveformBuffer,
    systemState,
    targetIntent,
    sendCommand,
  } = useBCISocket();

  // Flash key: incremented each time a new high-confidence event fires
  const [flashKey, setFlashKey] = useState(0);
  const prevTimestampRef = useRef('');

  useEffect(() => {
    if (
      latest?.high_confidence &&
      latest.timestamp !== prevTimestampRef.current
    ) {
      prevTimestampRef.current = latest.timestamp;
      setFlashKey(k => k + 1);
    }
  }, [latest]);

  const handleStart    = () => sendCommand({ command: 'START' });
  const handlePause    = () => sendCommand({ command: 'PAUSE' });
  const handleReset    = () => sendCommand({ command: 'RESET' });
  const handleTarget   = (t: IntentTarget) => sendCommand({ command: 'SET_TARGET', value: t });

  return (
    <div className="min-h-screen bg-[#0a0f1e] p-5 font-sans text-slate-100">
      {/* ── Header ── */}
      <header className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-white">
            BCI Verification Dashboard
          </h1>
          <p className="mt-0.5 text-sm text-slate-500">
            Real-time inner speech decoding — YES&thinsp;/&thinsp;NO intent monitor
          </p>
        </div>
        <ConnectionBadge status={status} />
      </header>

      <div className="mx-auto max-w-4xl space-y-4">
        {/* ── Intent + Confidence ── */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/50 py-8">
            <IntentDisplay
              predictedClass={latest?.predicted_class ?? null}
              highConfidence={latest?.high_confidence ?? false}
              flashKey={flashKey}
            />
          </div>
          <div className="flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/50 py-8">
            <ConfidenceGauge confidence={latest?.confidence ?? 0} />
          </div>
        </div>

        {/* ── Waveform ── */}
        <WaveformChart buffer={waveformBuffer} />

        {/* ── Stats ── */}
        <StatsRow
          accuracy={latest?.overall_accuracy ?? 0}
          epochCount={latest?.epoch_count ?? 0}
          targetIntent={targetIntent}
          groundTruth={latest?.ground_truth ?? null}
        />

        {/* ── Controls ── */}
        <ControlPanel
          systemState={systemState}
          targetIntent={targetIntent}
          onStart={handleStart}
          onPause={handlePause}
          onReset={handleReset}
          onSetTarget={handleTarget}
        />

        {/* ── Footer ── */}
        <footer className="pb-2 text-center text-[10px] text-slate-700">
          LockedIn Communicator · ASE 2026 ·{' '}
          <span className="font-mono">ws://localhost:8765</span>
        </footer>
      </div>
    </div>
  );
}
