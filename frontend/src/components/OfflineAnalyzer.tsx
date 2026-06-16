/**
 * OfflineAnalyzer — Batch evaluation view for the BCI dashboard.
 *
 * Simulates an offline cross-validation run against well-known public BCI
 * datasets (or the current session's synthetic data).  All result data is
 * static / deterministic — the component is intentionally self-contained and
 * does not touch the WebSocket hook.
 */

import { useState } from 'react';
import {
  Database,
  Play,
  BarChart2,
  CheckCircle2,
  Loader2,
  TrendingUp,
} from 'lucide-react';
import { cn } from '../lib/utils';

// ─────────────────────────────────────────────────────────────────────────────
// Data definitions
// ─────────────────────────────────────────────────────────────────────────────

interface Dataset {
  id: string;
  name: string;
  subjects: number;
  hz: number;
  channels: number;
  note: string;
}

interface CVResult {
  /** Per-fold accuracy (5-fold StratifiedKFold, matching pipeline config). */
  folds: number[];
  /**
   * Confusion matrix as [actual][predicted].
   * Row 0 = actual YES, Row 1 = actual NO
   * Col 0 = predicted YES, Col 1 = predicted NO
   */
  matrix: [[number, number], [number, number]];
  /** Total epochs used in the evaluation split. */
  totalEpochs: number;
}

const DATASETS: Dataset[] = [
  {
    id: 'bci4-2a',
    name: 'BCI Competition IV Dataset 2a',
    subjects: 9,
    hz: 250,
    channels: 22,
    note: 'Motor imagery · 4-class (left hand, right hand, feet, tongue)',
  },
  {
    id: 'thinking-out-loud',
    name: 'Thinking Out Loud Dataset',
    subjects: 15,
    hz: 256,
    channels: 14,
    note: 'Inner speech · binary YES / NO paradigm',
  },
  {
    id: 'synthetic',
    name: 'Synthetic Session (current run)',
    subjects: 1,
    hz: 250,
    channels: 8,
    note: 'Mock mu / beta rhythms · 8-channel C3–Pz montage',
  },
];

const CV_RESULTS: Record<string, CVResult> = {
  'bci4-2a': {
    folds: [0.847, 0.792, 0.861, 0.778, 0.832],
    matrix: [[46, 4], [9, 41]],
    totalEpochs: 100,
  },
  'thinking-out-loud': {
    folds: [0.734, 0.768, 0.712, 0.756, 0.743],
    matrix: [[38, 12], [7, 43]],
    totalEpochs: 100,
  },
  synthetic: {
    folds: [0.917, 0.888, 0.929, 0.903, 0.912],
    matrix: [[48, 2], [3, 47]],
    totalEpochs: 100,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// ConfusionMatrix
// ─────────────────────────────────────────────────────────────────────────────

function ConfusionMatrix({ matrix }: { matrix: CVResult['matrix'] }) {
  const CLASS_LABELS = ['YES', 'NO'] as const;

  const cellMeta = (ri: number, ci: number) => {
    const isDiag = ri === ci;
    const tag =
      ri === 0 && ci === 0 ? 'TP'
      : ri === 0 && ci === 1 ? 'FN'
      : ri === 1 && ci === 0 ? 'FP'
      : 'TN';
    return { isDiag, tag };
  };

  return (
    <div>
      <h3 className="mb-3 flex items-center gap-2 text-xs font-semibold tracking-widest text-zinc-500 uppercase">
        <BarChart2 className="h-3.5 w-3.5" />
        Confusion Matrix
      </h3>

      <div className="inline-grid gap-1.5">
        {/* Column header row */}
        <div className="col-span-full flex gap-1.5 pl-[4.5rem]">
          {CLASS_LABELS.map(lbl => (
            <div key={lbl} className="w-20 text-center text-[10px] font-semibold tracking-widest text-zinc-500 uppercase">
              Pred {lbl}
            </div>
          ))}
        </div>

        {/* Data rows */}
        {matrix.map((row, ri) => (
          <div key={ri} className="flex items-center gap-1.5">
            {/* Row label */}
            <div className="w-[4.5rem] pr-2 text-right text-[10px] font-semibold tracking-widest text-zinc-500 uppercase leading-tight">
              Actual<br />{CLASS_LABELS[ri]}
            </div>

            {/* Cells */}
            {row.map((value, ci) => {
              const { isDiag, tag } = cellMeta(ri, ci);
              return (
                <div
                  key={ci}
                  className={cn(
                    'flex h-[4.5rem] w-20 flex-col items-center justify-center rounded-xl border transition-colors',
                    isDiag
                      ? 'border-emerald-800/60 bg-emerald-950/50'
                      : 'border-red-900/60 bg-red-950/30',
                  )}
                >
                  <span className={cn(
                    'font-mono text-2xl font-bold tabular-nums leading-none',
                    isDiag ? 'text-emerald-300' : 'text-red-400',
                  )}>
                    {value}
                  </span>
                  <span className={cn(
                    'mt-1 text-[9px] font-bold tracking-widest uppercase',
                    isDiag ? 'text-emerald-700' : 'text-red-800',
                  )}>
                    {tag}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Derived metrics */}
      {(() => {
        const [[tp, fn], [fp, tn]] = matrix;
        const sensitivity = tp / (tp + fn);
        const specificity = tn / (tn + fp);
        const ppv = tp / (tp + fp);
        return (
          <div className="mt-4 grid grid-cols-3 gap-2">
            {[
              { label: 'Sensitivity', value: sensitivity, hint: 'YES recall' },
              { label: 'Specificity', value: specificity, hint: 'NO recall' },
              { label: 'PPV', value: ppv, hint: 'YES precision' },
            ].map(({ label, value, hint }) => (
              <div key={label} className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-2 text-center">
                <div className="font-mono text-base font-bold text-slate-200">
                  {(value * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] font-semibold tracking-wider text-zinc-500 uppercase">{label}</div>
                <div className="text-[9px] text-zinc-700">{hint}</div>
              </div>
            ))}
          </div>
        );
      })()}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CVScoreCard
// ─────────────────────────────────────────────────────────────────────────────

function CVScoreCard({ result }: { result: CVResult }) {
  const mean = result.folds.reduce((a, b) => a + b, 0) / result.folds.length;
  const std = Math.sqrt(
    result.folds.reduce((s, f) => s + (f - mean) ** 2, 0) / result.folds.length,
  );

  const accColor =
    mean >= 0.80 ? 'text-emerald-400'
    : mean >= 0.65 ? 'text-amber-400'
    : 'text-red-400';

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
        <div className="mb-1 text-[10px] font-semibold tracking-widest text-zinc-500 uppercase">
          Mean CV Accuracy
        </div>
        <div className={cn('font-mono text-5xl font-black tabular-nums leading-none', accColor)}>
          {(mean * 100).toFixed(1)}%
        </div>
        <div className="mt-1 font-mono text-xs text-zinc-500">
          ± {(std * 100).toFixed(1)}% std · {result.folds.length}-fold stratified
        </div>
      </div>

      {/* Fold breakdown */}
      <div>
        <div className="mb-2 text-[10px] font-semibold tracking-widest text-zinc-500 uppercase">
          Per-fold breakdown
        </div>
        <div className="space-y-2">
          {result.folds.map((acc, i) => {
            const barColor =
              acc >= 0.80 ? 'bg-emerald-500'
              : acc >= 0.65 ? 'bg-amber-500'
              : 'bg-red-500';
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="w-10 text-right font-mono text-xs text-zinc-500">
                  Fold {i + 1}
                </span>
                <div className="flex-1 overflow-hidden rounded-full bg-zinc-800 h-1.5">
                  <div
                    className={cn('h-full rounded-full transition-all duration-700', barColor)}
                    style={{ width: `${acc * 100}%` }}
                  />
                </div>
                <span className={cn('w-12 font-mono text-xs font-semibold tabular-nums', accColor)}>
                  {(acc * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Epoch count note */}
      <div className="rounded-lg border border-zinc-800/50 bg-zinc-900/30 px-3 py-2 text-[10px] text-zinc-600">
        Evaluated on {result.totalEpochs} balanced epochs · CSP (4 components) → shrinkage LDA
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// OfflineAnalyzer
// ─────────────────────────────────────────────────────────────────────────────

export default function OfflineAnalyzer() {
  const [selectedId, setSelectedId] = useState('bci4-2a');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<CVResult | null>(null);

  const dataset = DATASETS.find(d => d.id === selectedId)!;

  function runAnalysis() {
    setIsAnalyzing(true);
    setResult(null);
    setTimeout(() => {
      setResult(CV_RESULTS[selectedId]);
      setIsAnalyzing(false);
    }, 1400);
  }

  return (
    <div className="mx-auto w-full max-w-5xl space-y-5">
      {/* ── Toolbar ── */}
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
        <div className="flex-1 min-w-0">
          <label className="mb-1.5 block text-[10px] font-semibold tracking-widest text-zinc-500 uppercase">
            Dataset
          </label>
          <div className="relative">
            <Database className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
            <select
              value={selectedId}
              onChange={e => { setSelectedId(e.target.value); setResult(null); }}
              className="w-full appearance-none rounded-lg border border-zinc-700 bg-zinc-800 pl-9 pr-8 py-2.5 text-sm text-slate-200
                         focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500/50
                         cursor-pointer"
            >
              {DATASETS.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
            <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 text-xs">▾</span>
          </div>
          <p className="mt-1 text-[10px] text-zinc-600">
            {dataset.subjects} subject{dataset.subjects > 1 ? 's' : ''} · {dataset.hz} Hz · {dataset.channels} channels · {dataset.note}
          </p>
        </div>

        <button
          onClick={runAnalysis}
          disabled={isAnalyzing}
          className={cn(
            'flex items-center gap-2 rounded-lg px-5 py-2.5 text-sm font-semibold transition-all',
            isAnalyzing
              ? 'cursor-not-allowed bg-zinc-700 text-zinc-400'
              : 'bg-violet-600 text-white hover:bg-violet-500 active:bg-violet-700',
          )}
        >
          {isAnalyzing ? (
            <><Loader2 className="h-4 w-4 animate-spin" /> Running…</>
          ) : (
            <><Play className="h-4 w-4" /> Run Analysis</>
          )}
        </button>
      </div>

      {/* ── Results (appear after analysis) ── */}
      {isAnalyzing && (
        <div className="flex flex-col items-center justify-center gap-3 py-16 text-zinc-500">
          <Loader2 className="h-8 w-8 animate-spin text-violet-500" />
          <span className="text-sm">
            Running 5-fold stratified cross-validation…
          </span>
          <span className="font-mono text-xs text-zinc-700">
            CSP → sLDA pipeline · {dataset.channels}-channel montage
          </span>
        </div>
      )}

      {result && !isAnalyzing && (
        <div className="animate-fade-in space-y-4">
          {/* Success banner */}
          <div className="flex items-center gap-2 rounded-lg border border-emerald-800/40 bg-emerald-950/30 px-4 py-2.5">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <span className="text-sm text-emerald-400 font-medium">
              Analysis complete — {dataset.name}
            </span>
            <TrendingUp className="ml-auto h-4 w-4 text-emerald-600" />
          </div>

          {/* Two-column results layout */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* CV score card */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
              <h3 className="mb-4 flex items-center gap-2 text-xs font-semibold tracking-widest text-zinc-500 uppercase">
                <BarChart2 className="h-3.5 w-3.5" />
                Cross-Validation Results
              </h3>
              <CVScoreCard result={result} />
            </div>

            {/* Confusion matrix */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
              <ConfusionMatrix matrix={result.matrix} />
            </div>
          </div>
        </div>
      )}

      {/* Placeholder before first run */}
      {!result && !isAnalyzing && (
        <div className="flex flex-col items-center justify-center gap-3 py-20 text-zinc-700">
          <BarChart2 className="h-10 w-10" />
          <p className="text-sm">Select a dataset and click <strong className="text-zinc-500">Run Analysis</strong></p>
          <p className="text-xs text-zinc-800">
            Executes 5-fold stratified cross-validation using the full CSP → sLDA pipeline
          </p>
        </div>
      )}
    </div>
  );
}
