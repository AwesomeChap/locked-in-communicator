/**
 * Dashboard.tsx — Premium 3-column BCI workspace.
 *
 * Layout
 * ──────────────────────────────────────────────────────────────────────────
 *  TopBar     │ Wordmark · Segmented ModeToggle · ConnectionBadge
 * ────────────┼───────────────────────────────────────────────────────────
 *  LeftPanel  │ CenterPanel                      │ RightSidebar (mode-conditional)
 *  (192 px)   │ (flex-1)                         │ (276 px)
 *             │   Intent Card  │ Confidence Gauge│  ─ 'online'  → OnlineSidebar
 *  Metrics:   │   ─────────────────────────────  │    checklist + controls
 *  · Accuracy │   WaveformChart (full width)      │  ─ 'offline' → OfflineSidebar
 *  · Epochs   │                                  │    dataset picker + confusion matrix
 *  · Target   │                                  │
 * ──────────────────────────────────────────────────────────────────────────
 *
 * The useBCISocket hook stays fully active in both modes, so the left and
 * center panels always reflect live telemetry.
 */

import { useEffect, useRef, useState } from 'react';
import {
  LineChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Zap,
  BarChart2,
  Play,
  Pause,
  RotateCcw,
  Loader2,
  Wifi,
  WifiOff,
  RefreshCw,
  Database,
  CheckSquare,
  Square,
  ShieldCheck,
} from 'lucide-react';
import { useBCISocket } from '../hooks/useBCISocket';
import type {
  ConnectionStatus,
  IntentClass,
  SystemState,
} from '../types/bci';
import { cn } from '../lib/utils';

// ─────────────────────────────────────────────────────────────────────────────
// Types & static data
// ─────────────────────────────────────────────────────────────────────────────

type AppMode = 'online' | 'offline';

const DATASETS = [
  { id: 'bci4-2a-s1',       name: 'BCI Competition IV 2a – Session 1', hz: 250, ch: 22 },
  { id: 'bci4-2a-s2',       name: 'BCI Competition IV 2a – Session 2', hz: 250, ch: 22 },
  { id: 'thinking-out-loud', name: 'Thinking Out Loud Dataset',         hz: 256, ch: 14 },
  { id: 'synthetic',         name: 'Synthetic Session (current run)',   hz: 250, ch: 8  },
] as const;

type DatasetId = (typeof DATASETS)[number]['id'];

const CV_ACCURACY: Record<DatasetId, number> = {
  'bci4-2a-s1':       0.870,
  'bci4-2a-s2':       0.854,
  'thinking-out-loud': 0.743,
  synthetic:           0.912,
};

const CONFUSION: Record<DatasetId, [[number, number], [number, number]]> = {
  'bci4-2a-s1':       [[46, 4],  [9,  41]],
  'bci4-2a-s2':       [[44, 6],  [7,  43]],
  'thinking-out-loud': [[38, 12], [7,  43]],
  synthetic:           [[48, 2],  [3,  47]],
};

const CV_EPOCHS: Record<DatasetId, number> = {
  'bci4-2a-s1':        100,
  'bci4-2a-s2':        100,
  'thinking-out-loud': 100,
  synthetic:           100,
};

/** Single classified epoch produced during offline playback. */
interface OfflineEpoch {
  predictedClass: IntentClass;
  groundTruth:    IntentClass;
  confidence:     number;
  highConfidence: boolean;
  /** Downsampled EEG waveform snapshot (120 pts) for the chart. */
  waveform:       number[];
}

/** Result shape shared between OfflineSidebar and the Dashboard root. */
interface OfflineResult {
  acc:        number;
  epochCount: number;
  matrix:     [[number, number], [number, number]];
  /** Full epoch sequence — played back at 2 Hz to drive the center panels. */
  epochs:     OfflineEpoch[];
}

// ─── Offline playback helpers ────────────────────────────────────────────────

/** Synthesise a 120-point EEG waveform consistent with the epoch class. */
function syntheticWaveform(cls: IntentClass): number[] {
  const freq  = cls === 'YES' ? 10 : 20;          // mu (10 Hz) vs beta (20 Hz)
  const phase = Math.random() * Math.PI * 2;
  const out: number[] = [];
  for (let i = 0; i < 120; i++) {
    const t = i / 250;
    out.push(
      2.0 * Math.sin(2 * Math.PI * freq * t + phase) +
      (Math.random() - 0.5) * 1.3,
    );
  }
  return out;
}

/**
 * Build a shuffled epoch sequence consistent with the confusion matrix.
 * Correct predictions (TP/TN) get high confidence (≥ 0.72).
 * Errors (FN/FP) get borderline confidence (0.50–0.69).
 */
function generateEpochSequence(
  matrix: [[number, number], [number, number]],
): OfflineEpoch[] {
  const [[tp, fn], [fp, tn]] = matrix;
  const THRESHOLD = 0.70;
  const epochs: OfflineEpoch[] = [];

  const push = (
    gt: IntentClass, pred: IntentClass,
    cMin: number, cMax: number,
  ) => {
    const confidence = cMin + Math.random() * (cMax - cMin);
    epochs.push({
      groundTruth:    gt,
      predictedClass: pred,
      confidence,
      highConfidence: confidence >= THRESHOLD,
      waveform:       syntheticWaveform(gt),
    });
  };

  for (let i = 0; i < tp; i++) push('YES', 'YES', 0.72, 0.97);
  for (let i = 0; i < fn; i++) push('YES', 'NO',  0.50, 0.69);
  for (let i = 0; i < fp; i++) push('NO',  'YES', 0.50, 0.69);
  for (let i = 0; i < tn; i++) push('NO',  'NO',  0.72, 0.97);

  // Fisher-Yates shuffle
  for (let i = epochs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [epochs[i], epochs[j]] = [epochs[j], epochs[i]];
  }
  return epochs;
}

const CHECKLIST = [
  { id: 'gnd', label: 'Verify Ground Reference',            note: 'DRL/CMS electrode seated and secured'     },
  { id: 'c3',  label: 'Check C3 Node (Left Motor Strip)',   note: 'Impedance < 10 kΩ — primary YES signal'   },
  { id: 'c4',  label: 'Check C4 Node (Right Motor Strip)',  note: 'Impedance < 10 kΩ — primary NO signal'    },
  { id: 'cz',  label: 'Check Cz Vertex',                    note: 'Impedance < 10 kΩ — midline CSP reference' },
] as const;

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionBadge
// ─────────────────────────────────────────────────────────────────────────────

function ConnectionBadge({ status }: { status: ConnectionStatus }) {
  const Icon =
    status === 'CONNECTED'    ? Wifi
    : status === 'RECONNECTING' ? RefreshCw
    : WifiOff;

  return (
    <div className={cn(
      'flex items-center gap-1.5 rounded-full border px-2.5 py-1',
      'text-[9px] font-bold tracking-widest uppercase select-none',
      status === 'CONNECTED'    && 'border-emerald-800 bg-emerald-950/60 text-emerald-400',
      status === 'RECONNECTING' && 'border-amber-800  bg-amber-950/60  text-amber-400',
      status === 'DISCONNECTED' && 'border-red-900    bg-red-950/60    text-red-500',
    )}>
      <Icon className={cn(
        'h-3 w-3 shrink-0',
        status === 'CONNECTED'    && 'animate-pulse',
        status === 'RECONNECTING' && 'animate-spin',
      )} />
      {status}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ModeToggle — iOS-style segmented control
// ─────────────────────────────────────────────────────────────────────────────

function ModeToggle({ mode, onChange }: { mode: AppMode; onChange: (m: AppMode) => void }) {
  return (
    <div className="flex rounded-lg border border-zinc-700/70 bg-zinc-900 p-0.5 shadow-inner">
      {(
        [
          { id: 'online',  label: 'Live Stream',      Icon: Zap       },
          { id: 'offline', label: 'Dataset Analysis', Icon: BarChart2 },
        ] as const
      ).map(({ id, label, Icon }) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={cn(
            'flex items-center gap-1.5 rounded-md px-4 py-1.5',
            'text-xs font-semibold transition-all duration-200',
            mode === id
              ? 'bg-zinc-100 text-zinc-950 shadow-sm'
              : 'text-zinc-500 hover:text-zinc-300',
          )}
        >
          <Icon className="h-3.5 w-3.5 shrink-0" />
          {label}
        </button>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TopBar
// ─────────────────────────────────────────────────────────────────────────────

function TopBar({
  mode,
  onModeChange,
  status,
}: {
  mode: AppMode;
  onModeChange: (m: AppMode) => void;
  status: ConnectionStatus;
}) {
  return (
    <header className="sticky top-0 z-20 flex items-center border-b border-zinc-800/60 bg-zinc-950/95 px-5 py-3 backdrop-blur-sm">
      {/* Wordmark */}
      <div className="flex w-48 shrink-0 items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600">
          <Zap className="h-4 w-4 text-white" />
        </div>
        <div>
          <div className="text-sm font-bold leading-none text-zinc-100">LockedIn BCI</div>
          <div className="mt-0.5 text-[9px] font-semibold tracking-widest text-zinc-600 uppercase">
            ASE 2026
          </div>
        </div>
      </div>

      {/* Centred mode toggle */}
      <div className="flex flex-1 justify-center">
        <ModeToggle mode={mode} onChange={onModeChange} />
      </div>

      {/* Connection badge */}
      <div className="flex w-48 shrink-0 justify-end">
        <ConnectionBadge status={status} />
      </div>
    </header>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// IntentCard — large YES/NO with flash + continuous glow on high confidence
// ─────────────────────────────────────────────────────────────────────────────

interface IntentCardProps {
  predictedClass: IntentClass | null;
  flashKey:       number;
  groundTruth:    IntentClass | null;
  mode:           AppMode;
}

function IntentCard({ predictedClass, flashKey, groundTruth, mode }: IntentCardProps) {
  const isOffline = mode === 'offline';
  const hasMatch  = predictedClass !== null && groundTruth !== null;
  const isCorrect = hasMatch && predictedClass === groundTruth;

  if (isOffline) {
    return (
      /* ── Offline: clean split ── */
      <div className="flex h-full rounded-xl border border-zinc-800/60 bg-zinc-900/50">

        {/* Shared label row + value row via CSS grid columns */}
        {/* Use a single-row grid so all labels share the same baseline */}
        {/* Fixed centre column width prevents layout shift between "correct" and "error" */}
        <div className="grid w-full" style={{ gridTemplateColumns: '1fr 80px 1fr' }}>

          {/* ── Row 1: labels ── */}
          {/* ── Row 1: labels (fixed top position) ── */}
          <div className="flex items-end justify-center pb-2 pt-5">
            <span className="text-[9px] font-semibold tracking-[0.18em] text-zinc-600 uppercase">
              Predicted
            </span>
          </div>
          <div className="flex items-end justify-center pb-2 pt-5">
            <span className={cn(
              'whitespace-nowrap text-[9px] font-semibold tracking-widest uppercase transition-colors duration-300',
              !hasMatch               && 'text-transparent',
              hasMatch && isCorrect   && 'text-emerald-400',
              hasMatch && !isCorrect  && 'text-red-400',
            )}>
              {isCorrect ? 'correct' : 'error'}
            </span>
          </div>
          <div className="flex items-end justify-center pb-2 pt-5">
            <span className="text-[9px] font-semibold tracking-[0.18em] text-zinc-600 uppercase">
              Expected
            </span>
          </div>

          {/* ── Row 2: values ── */}
          {/* Predicted — bright white */}
          <div key={flashKey} className="flex items-center justify-center pb-5 pt-3">
            <span className={cn(
              'font-mono text-5xl font-semibold tabular-nums leading-none transition-colors duration-300',
              predictedClass === null && 'text-zinc-700',
              predictedClass !== null && 'text-white',
            )}>
              {predictedClass ?? '—'}
            </span>
          </div>

          {/* Match symbol — green / red */}
          <div className="flex items-center justify-center pb-5 pt-3">
            <span className={cn(
              'font-mono text-5xl font-semibold leading-none transition-colors duration-300',
              !hasMatch               && 'text-zinc-800',
              hasMatch && isCorrect   && 'text-emerald-400',
              hasMatch && !isCorrect  && 'text-red-400',
            )}>
              {!hasMatch ? '·' : isCorrect ? '=' : '≠'}
            </span>
          </div>

          {/* Expected — slightly muted white */}
          <div className="flex items-center justify-center pb-5 pt-3">
            <span className={cn(
              'font-mono text-5xl font-semibold tabular-nums leading-none transition-colors duration-300',
              groundTruth === null && 'text-zinc-700',
              groundTruth !== null && 'text-zinc-400',
            )}>
              {groundTruth ?? '—'}
            </span>
          </div>

        </div>
      </div>
    );
  }

  /* ── Online: clean text + accent line ── */
  return (
    <div className="flex h-full flex-col rounded-xl border border-zinc-800/60 bg-zinc-900/50">
      <div className="flex flex-1 flex-col items-center justify-center gap-3 px-8 pt-5 pb-4">
        <span className="text-[9px] font-semibold tracking-[0.18em] text-zinc-600 uppercase">
          Decoded Intent
        </span>

        <span
          className={cn(
            'font-mono text-7xl font-semibold tabular-nums leading-none transition-colors duration-300',
            predictedClass === null  && 'text-zinc-700',
            predictedClass === 'YES' && 'text-emerald-400',
            predictedClass === 'NO'  && 'text-red-400',
          )}
        >
          {predictedClass ?? '—'}
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfidenceGauge — SVG half-arc radial meter
// ─────────────────────────────────────────────────────────────────────────────

function ConfidenceGauge({ confidence }: { confidence: number }) {
  // The arc path spans a chord of 132 px (x: 11→143) with a specified r=60.
  // SVG scales the radius up when chord > 2r: r_eff = chord/2 = 66.
  // Using the wrong R would make ARC_LEN shorter than the actual path, leaving
  // the right end of the track exposed regardless of fill percentage.
  const R       = 66;
  const ARC_LEN = Math.PI * R; // ≈ 207.3 — matches actual SVG arc length

  const clamped = Math.max(0, Math.min(1, confidence));
  const offset  = ARC_LEN * (1 - clamped);
  const pct     = Math.round(clamped * 100);
  const color   = clamped >= 0.7 ? '#10b981' : clamped >= 0.5 ? '#f59e0b' : '#ef4444';

  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 rounded-xl border border-zinc-800/60 bg-zinc-900/50 py-8">
      <span className="text-[9px] font-bold tracking-widest text-zinc-600 uppercase">
        Confidence
      </span>

      <svg viewBox="0 0 154 88" className="w-36" aria-label={`${pct}% confidence`}>
        {/* Track */}
        <path
          d="M 11 80 A 60 60 0 0 1 143 80"
          stroke="#27272a"
          strokeWidth="12"
          fill="none"
          strokeLinecap="round"
        />
        {/* Progress — hidden below 1% to avoid rogue linecap at right end */}
        {clamped >= 0.01 && (
          <path
            d="M 11 80 A 60 60 0 0 1 143 80"
            stroke={color}
            strokeWidth="12"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={`${ARC_LEN} ${ARC_LEN}`}
            strokeDashoffset={offset}
            style={{
              transition:
                'stroke-dashoffset 0.45s cubic-bezier(0.4,0,0.2,1), stroke 0.3s ease',
            }}
          />
        )}
        <text
          x="77"
          y="69"
          textAnchor="middle"
          fill="#f4f4f5"
          fontSize="20"
          fontWeight="700"
          fontFamily="'JetBrains Mono', monospace"
        >
          {pct}%
        </text>
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// WaveformChart — rolling Recharts line for channel C3
// ─────────────────────────────────────────────────────────────────────────────

function WaveformChart({ buffer, epochCount }: { buffer: number[]; epochCount?: number }) {
  const data = buffer.map((v, i) => ({ t: i, v }));

  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/50 p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[9px] font-bold tracking-widest text-zinc-600 uppercase">
          EEG Signal — Ch. C3
        </span>
        <div className="flex items-center gap-3">
          {epochCount !== undefined && epochCount > 0 && (
            <span className="font-mono text-[9px] text-zinc-300">
              {epochCount.toLocaleString()} epochs
            </span>
          )}
          <span className="font-mono text-[9px] text-zinc-500">250 Hz · 8–30 Hz bandpass</span>
        </div>
      </div>
      <div className="h-48">
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
          <div className="flex h-full items-center justify-center gap-2 text-xs text-zinc-500">
            <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-700 border-t-blue-500" />
            Awaiting signal stream…
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineSidebar — signal verification checklist + acquisition controls
// ─────────────────────────────────────────────────────────────────────────────

interface OnlineSidebarProps {
  systemState: SystemState;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
}

function OnlineSidebar({
  systemState,
  onStart,
  onPause,
  onReset,
}: OnlineSidebarProps) {
  const [checked, setChecked] = useState<Set<string>>(new Set());
  // lslRunning tracks whether a hardware acquisition session is active.
  // lslConnected will be driven by a backend LSL-status message once supported;
  // for now it is always false (button shown disabled until device is present).
  const [lslRunning, setLslRunning] = useState(false);
  const lslConnected = false; // TODO: wire from backend LSL status message

  const toggle = (id: string) =>
    setChecked(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const allChecked = checked.size === CHECKLIST.length;
  const isRunning  = systemState === 'RUNNING';
  const isFitting  = systemState === 'FITTING';

  return (
    <div className="flex flex-col gap-3">
      {/* ── Signal verification checklist ── */}
      <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/50 p-4">
        <div className="mb-3 flex items-center gap-2.5 px-2">
          <ShieldCheck className={cn(
            'mt-0.5 h-4 w-4 shrink-0 transition-colors',
            allChecked ? 'text-emerald-400' : 'text-zinc-600',
          )} />
          <span className="text-[10px] font-bold tracking-widest text-zinc-300 uppercase">
            10-20 System Signal Verification
          </span>
        </div>

        <div className="space-y-1">
          {CHECKLIST.map(({ id, label, note }) => {
            const active = checked.has(id);
            return (
              <button
                key={id}
                onClick={() => toggle(id)}
                className="flex w-full items-start gap-2.5 rounded-lg px-2 py-2 text-left transition-colors hover:bg-zinc-800/50"
              >
                {active
                  ? <CheckSquare className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                  : <Square      className="mt-0.5 h-4 w-4 shrink-0 text-zinc-600" />
                }
                <div>
                  <div className={cn(
                    'text-xs font-semibold leading-snug transition-colors',
                    active ? 'text-zinc-500 line-through decoration-zinc-600' : 'text-zinc-200',
                  )}>
                    {label}
                  </div>
                  <div className="mt-0.5 text-[9px] leading-snug text-zinc-500">{note}</div>
                </div>
              </button>
            );
          })}
        </div>

        {allChecked && (
          <div className="mt-3 rounded-lg border border-emerald-800/40 bg-emerald-950/30 px-3 py-2">
            <span className="text-[9px] font-semibold text-emerald-400">
              ✓ All nodes verified — ready to acquire
            </span>
          </div>
        )}
      </div>

      {/* ── Live EEG acquisition ── */}
      <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/50 p-4">
        <div className="mb-3 text-[9px] font-bold tracking-widest text-zinc-600 uppercase">
          Live Acquisition
        </div>

        {/* Device status */}
        <div className="mb-3 flex items-center gap-2">
          <span className={cn(
            'h-2 w-2 shrink-0 rounded-full',
            lslConnected ? 'bg-emerald-400 shadow-[0_0_6px_#34d399]' : 'bg-zinc-700',
          )} />
          <span className="text-[10px] text-zinc-500">
            {lslConnected ? 'EEG device ready' : 'No device detected'}
          </span>
        </div>

        {lslRunning ? (
          <div className="flex gap-2">
            <div className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border
                            border-emerald-800/50 bg-emerald-950/30 py-2">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
              <span className="text-[10px] font-semibold text-emerald-400">Recording</span>
            </div>
            <button
              onClick={() => setLslRunning(false)}
              className="flex items-center justify-center gap-1.5 rounded-lg px-4 py-2
                         text-xs font-semibold bg-red-900/60 text-red-400 border border-red-800/50
                         hover:bg-red-900/80 transition-colors"
            >
              End
            </button>
          </div>
        ) : (
          <button
            disabled={!lslConnected}
            onClick={() => setLslRunning(true)}
            className={cn(
              'flex w-full items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-colors',
              lslConnected
                ? 'bg-violet-600 text-white hover:bg-violet-500 active:bg-violet-700'
                : 'cursor-not-allowed bg-zinc-800/60 text-zinc-600',
            )}
          >
            <Play className="h-3.5 w-3.5" />
            Run
          </button>
        )}
      </div>

      {/* ── Simulation ── */}
      <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/50 p-4">
        <div className="mb-3 text-[9px] font-bold tracking-widest text-zinc-600 uppercase">
          Simulation
        </div>

        {/* Start / Pause */}
        {isFitting ? (
          <button disabled className="flex w-full items-center justify-center gap-2 rounded-lg
                                      py-2 text-xs font-semibold bg-zinc-800 text-zinc-500 cursor-not-allowed">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Fitting model…
          </button>
        ) : isRunning ? (
          <div className="flex gap-2">
            <button
              onClick={onPause}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2
                         text-xs font-semibold bg-amber-600 text-white hover:bg-amber-500 transition-colors"
            >
              <Pause className="h-3.5 w-3.5" /> Pause
            </button>
            <button
              onClick={onReset}
              title="Reset"
              className="flex items-center justify-center rounded-lg px-3 py-2
                         border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={onStart}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2
                         text-xs font-semibold bg-violet-600 text-white hover:bg-violet-500 active:bg-violet-700 transition-colors"
            >
              <Play className="h-3.5 w-3.5" /> Start
            </button>
            <button
              onClick={onReset}
              title="Reset"
              className="flex items-center justify-center rounded-lg px-3 py-2
                         border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// OfflineSidebar — dataset selector + confusion matrix
// ─────────────────────────────────────────────────────────────────────────────

interface OfflineSidebarProps {
  onResult:        (r: OfflineResult | null) => void;
  isPaused:        boolean;
  isPlaybackOn:    boolean;
  onPause:         () => void;
  onResume:        () => void;
  /** Index of the epoch currently shown in the center panels (−1 = none yet). */
  currentEpochIdx: number;
}

function OfflineSidebar({
  onResult, isPaused, isPlaybackOn, onPause, onResume, currentEpochIdx,
}: OfflineSidebarProps) {
  const [selectedId, setSelectedId] = useState<DatasetId>('bci4-2a-s1');
  const [isRunning,  setIsRunning]  = useState(false);
  const [result, setResult]         = useState<OfflineResult | null>(null);

  const dataset = DATASETS.find(d => d.id === selectedId)!;

  // ── Running confusion matrix (updates every tick) ──────────────────────
  const seenCount = result ? Math.min(currentEpochIdx + 1, result.epochs.length) : 0;

  const runningMatrix = (() => {
    if (!result || seenCount === 0) return null;
    const m: [[number, number], [number, number]] = [[0, 0], [0, 0]];
    for (let i = 0; i < seenCount; i++) {
      const e  = result.epochs[i];
      const ri = e.groundTruth    === 'YES' ? 0 : 1;
      const ci = e.predictedClass === 'YES' ? 0 : 1;
      m[ri][ci]++;
    }
    return m;
  })();

  const runningAcc = runningMatrix
    ? (runningMatrix[0][0] + runningMatrix[1][1]) / seenCount
    : 0;

  const runningSensitivity = runningMatrix && (runningMatrix[0][0] + runningMatrix[0][1]) > 0
    ? runningMatrix[0][0] / (runningMatrix[0][0] + runningMatrix[0][1])
    : 0;

  const runningSpecificity = runningMatrix && (runningMatrix[1][1] + runningMatrix[1][0]) > 0
    ? runningMatrix[1][1] / (runningMatrix[1][1] + runningMatrix[1][0])
    : 0;

  function runAnalysis() {
    setIsRunning(true);
    setResult(null);
    onResult(null);
    setTimeout(() => {
      const r: OfflineResult = {
        acc:        CV_ACCURACY[selectedId],
        epochCount: CV_EPOCHS[selectedId],
        matrix:     CONFUSION[selectedId],
        epochs:     generateEpochSequence(CONFUSION[selectedId]),
      };
      setResult(r);
      onResult(r);
      setIsRunning(false);
    }, 1200);
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Dataset selector */}
      <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/50 p-4">
        <div className="mb-2 text-[9px] font-bold tracking-widest text-zinc-600 uppercase">
          Dataset
        </div>

        <div className="relative">
          <Database className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
          <select
            value={selectedId}
            onChange={e => {
              setSelectedId(e.target.value as DatasetId);
              setResult(null);
              onResult(null);
            }}
            className="w-full appearance-none rounded-lg border border-zinc-700 bg-zinc-800
                       pl-8 pr-6 py-2 text-xs text-zinc-200
                       focus:border-blue-500 focus:outline-none cursor-pointer"
          >
            {DATASETS.map(d => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
          <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-[9px] text-zinc-500">
            ▾
          </span>
        </div>

        <div className="mt-1.5 font-mono text-[10px] text-zinc-500">
          {dataset.ch} ch · {dataset.hz} Hz
        </div>


        {/* Control bar — state machine: idle → loading → playing ⇄ paused → done */}
        {isRunning ? (
          <button disabled className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg
                                      py-2 text-xs font-semibold bg-zinc-800 text-zinc-500 cursor-not-allowed">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Running…
          </button>
        ) : isPlaybackOn ? (
          /* Playback in progress — Pause or Resume */
          <div className="mt-3 flex gap-2">
            {isPaused ? (
              <button
                onClick={onResume}
                className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2
                           text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 transition-colors"
              >
                <Play className="h-3.5 w-3.5" /> Resume
              </button>
            ) : (
              <button
                onClick={onPause}
                className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2
                           text-xs font-semibold bg-amber-600 text-white hover:bg-amber-500 transition-colors"
              >
                <Pause className="h-3.5 w-3.5" /> Pause
              </button>
            )}
            <button
              onClick={runAnalysis}
              title="Re-run"
              className="flex items-center justify-center rounded-lg px-3 py-2
                         border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : result ? (
          /* Playback done — offer re-run */
          <button
            onClick={runAnalysis}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg py-2
                       text-xs font-semibold bg-zinc-700 text-zinc-200 hover:bg-zinc-600 transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Re-run
          </button>
        ) : (
          /* Initial state */
          <button
            onClick={runAnalysis}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg py-2
                       text-xs font-semibold bg-violet-600 text-white hover:bg-violet-500 active:bg-violet-700 transition-colors"
          >
              <Play className="h-3.5 w-3.5" /> Run
          </button>
        )}
      </div>

      {/* Confusion matrix — live-updating during playback */}
      {result && (
        <div className="animate-fade-in rounded-xl border border-zinc-800/60 bg-zinc-900/50 p-4">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[9px] font-bold tracking-widest text-zinc-300 uppercase">
              Confusion Matrix
            </span>
            <span className={cn(
              'font-mono text-xs font-bold tabular-nums transition-colors duration-300',
              runningAcc >= 0.80 ? 'text-emerald-400'
              : runningAcc >= 0.65 ? 'text-amber-400'
              : 'text-red-400',
            )}>
              {runningMatrix ? `${(runningAcc * 100).toFixed(1)}% acc` : '—'}
            </span>
          </div>

          {/* Epoch progress counter */}
          <div className="mb-3 font-mono text-[9px] text-zinc-500">
            {seenCount} / {result.epochCount} epochs
          </div>

          {/* Column headers */}
          <div className="mb-1.5 flex text-[9px] font-semibold tracking-wider text-zinc-500 uppercase">
            <div className="w-14" />
            <div className="flex-1 text-center">Pred YES</div>
            <div className="flex-1 text-center">Pred NO</div>
          </div>

          {/* 2×2 grid */}
          <div className="space-y-1.5">
            {(runningMatrix ?? [[0,0],[0,0]]).map((row, ri) => (
              <div key={ri} className="flex items-center gap-1.5">
                <div className="w-14 pr-1 text-right text-[9px] font-semibold uppercase text-zinc-500">
                  Act {ri === 0 ? 'YES' : 'NO'}
                </div>
                {row.map((val, ci) => {
                  const isDiag = ri === ci;
                  const tag =
                    ri === 0 && ci === 0 ? 'TP'
                    : ri === 0           ? 'FN'
                    : ci === 0           ? 'FP'
                    : 'TN';
                  return (
                    <div
                      key={ci}
                      className={cn(
                        'flex flex-1 flex-col items-center justify-center rounded-lg border py-5 transition-all duration-300',
                        isDiag
                          ? 'border-emerald-800/50 bg-emerald-950/40'
                          : 'border-red-900/40    bg-red-950/25',
                      )}
                    >
                      <span className={cn(
                        'font-mono text-3xl font-bold leading-none tabular-nums transition-all duration-300',
                        isDiag ? 'text-emerald-300' : 'text-red-400',
                      )}>
                        {val}
                      </span>
                      <span className={cn(
                        'mt-0.5 text-[8px] font-bold tracking-widest uppercase',
                        isDiag ? 'text-emerald-800' : 'text-red-800',
                      )}>
                        {tag}
                      </span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Sensitivity / Specificity */}
          <div className="mt-3 grid grid-cols-2 gap-1.5 text-center">
            {[
              { label: 'Sensitivity', val: runningSensitivity },
              { label: 'Specificity', val: runningSpecificity },
            ].map(({ label, val }) => (
              <div
                key={label}
                className="rounded-lg border border-zinc-800/50 bg-zinc-900/40 py-2"
              >
                <div className="font-mono text-sm font-bold tabular-nums text-zinc-200 transition-all duration-300">
                  {runningMatrix ? `${(val * 100).toFixed(1)}%` : '—'}
                </div>
                <div className="text-[8px] font-semibold tracking-wider text-zinc-500 uppercase">
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pre-run placeholder */}
      {!result && !isRunning && (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-zinc-800/40 py-10 text-zinc-500">
          <BarChart2 className="h-7 w-7" />
          <span className="text-[10px]">Run analysis to see results</span>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard — root
// ─────────────────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [mode, setMode] = useState<AppMode>('online');

  // Lifted offline analysis result + epoch sequence.
  const [offlineResult, setOfflineResult] = useState<OfflineResult | null>(null);

  // Clear offline result whenever the user leaves Dataset Analysis mode.
  const handleModeChange = (m: AppMode) => {
    setMode(m);
    if (m !== 'offline') setOfflineResult(null);
  };

  // The hook stays fully active in both modes (keeps the WS connection alive).
  const { status, latest, waveformBuffer, systemState, sendCommand } =
    useBCISocket();

  // ── Online flash key ──────────────────────────────────────────────────────
  const [flashKey, setFlashKey] = useState(0);
  const prevTsRef = useRef('');

  useEffect(() => {
    if (
      mode === 'online' &&
      latest?.high_confidence &&
      latest.timestamp !== prevTsRef.current
    ) {
      prevTsRef.current = latest.timestamp;
      setFlashKey(k => k + 1);
    }
  }, [latest, mode]);

  // ── Offline epoch playback ────────────────────────────────────────────────
  const [playbackIdx,     setPlaybackIdx]     = useState(0);
  const [offlineBuffer,   setOfflineBuffer]   = useState<number[]>([]);
  const [offlineFlashKey, setOfflineFlashKey] = useState(0);
  const [offlinePaused,   setOfflinePaused]   = useState(false);
  const prevOfflineIdxRef                     = useRef(-1);

  // Reset playback counters whenever a new analysis result arrives.
  useEffect(() => {
    setPlaybackIdx(0);
    setOfflineBuffer([]);
    prevOfflineIdxRef.current = -1;
    setOfflineFlashKey(0);
    setOfflinePaused(false);
  }, [offlineResult]);

  // Tick forward at 2 Hz (one epoch per 500 ms) — respects pause flag.
  useEffect(() => {
    if (mode !== 'offline' || !offlineResult || offlinePaused) return;
    if (playbackIdx >= offlineResult.epochs.length - 1) return;
    const t = setTimeout(() => setPlaybackIdx(i => i + 1), 500);
    return () => clearTimeout(t);
  }, [mode, offlineResult, playbackIdx, offlinePaused]);

  // Append the current epoch's waveform to the rolling offline chart buffer.
  useEffect(() => {
    if (mode !== 'offline' || !offlineResult) return;
    const epoch = offlineResult.epochs[playbackIdx];
    if (!epoch) return;
    setOfflineBuffer(prev => {
      const merged = prev.concat(epoch.waveform);
      return merged.length > 300 ? merged.slice(-300) : merged;
    });
  }, [mode, offlineResult, playbackIdx]);

  // Advance offline flash key when a high-confidence epoch is displayed.
  useEffect(() => {
    if (mode !== 'offline' || !offlineResult) return;
    const epoch = offlineResult.epochs[playbackIdx];
    if (epoch?.highConfidence && playbackIdx !== prevOfflineIdxRef.current) {
      prevOfflineIdxRef.current = playbackIdx;
      setOfflineFlashKey(k => k + 1);
    }
  }, [mode, offlineResult, playbackIdx]);

  // ── Display variable derivation ──────────────────────────────────────────
  const isOnline = mode === 'online';

  // Current offline epoch (null before first run or while in online mode).
  const currentEpoch = (!isOnline && offlineResult)
    ? (offlineResult.epochs[playbackIdx] ?? null)
    : null;

  // Playback is "running" while there are still epochs left to replay.
  const offlineRunning = !isOnline && offlineResult !== null
    && playbackIdx < offlineResult.epochs.length - 1;

  // Values fed to each shared panel — source switches on mode.
  const displayPredicted  = isOnline ? (latest?.predicted_class ?? null) : (currentEpoch?.predictedClass ?? null);
  const displayConfidence = isOnline ? (latest?.confidence ?? 0)         : (currentEpoch?.confidence ?? 0);
  const displayGroundTruth = isOnline ? (latest?.ground_truth ?? null)     : (currentEpoch?.groundTruth ?? null);
  const displayFlashKey    = isOnline ? flashKey : offlineFlashKey;
  const displayBuffer     = isOnline ? waveformBuffer : offlineBuffer;
  const displayEpochCount = isOnline ? (latest?.epoch_count ?? 0) : (offlineResult ? playbackIdx + 1 : 0);

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-50 font-sans">
      <TopBar mode={mode} onModeChange={handleModeChange} status={status} />

      <main className="flex flex-1 gap-4 overflow-hidden p-4">

        {/* ── Center: visualizers ── */}
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <div className="grid grid-cols-5 gap-3">
            <div className="col-span-3">
              <IntentCard
                predictedClass={displayPredicted}
                flashKey={displayFlashKey}
                groundTruth={displayGroundTruth}
                mode={mode}
              />
            </div>
            <div className="col-span-2">
              <ConfidenceGauge confidence={displayConfidence} />
            </div>
          </div>

          <WaveformChart buffer={displayBuffer} epochCount={displayEpochCount} />
        </div>

        {/* ── Right: mode-conditional sidebar ── */}
        <aside className="flex w-[380px] shrink-0 flex-col">
          {mode === 'online' ? (
            <OnlineSidebar
              systemState={systemState}
              onStart={() => sendCommand({ command: 'START' })}
              onPause={() => sendCommand({ command: 'PAUSE' })}
              onReset={() => sendCommand({ command: 'RESET' })}
            />
          ) : (
            <OfflineSidebar
              onResult={setOfflineResult}
              isPaused={offlinePaused}
              isPlaybackOn={offlineRunning || offlinePaused}
              onPause={() => setOfflinePaused(true)}
              onResume={() => setOfflinePaused(false)}
              currentEpochIdx={offlineResult ? playbackIdx : -1}
            />
          )}
        </aside>
      </main>

      <footer className="border-t border-zinc-800/60 py-2 text-center font-mono text-[9px] text-zinc-800">
        LockedIn Communicator · ASE 2026 · ws://localhost:8765
      </footer>
    </div>
  );
}
