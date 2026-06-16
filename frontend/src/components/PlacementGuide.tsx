/**
 * PlacementGuide — Collapsible side-drawer for the Online Acquisition view.
 *
 * Describes the 10-20 sensorimotor electrode montage used by the pipeline,
 * with emphasis on the C3/C4 lateralization that drives CSP class separation.
 */

import {
  ChevronRight,
  ChevronLeft,
  AlertTriangle,
  Circle,
  Radio,
} from 'lucide-react';
import { cn } from '../lib/utils';

// ─────────────────────────────────────────────────────────────────────────────
// Electrode node data
// ─────────────────────────────────────────────────────────────────────────────

interface ElectrodeInfo {
  id: string;
  label: string;
  region: string;
  hemisphere: 'left' | 'right' | 'midline';
  role: string;
  detail: string;
  isPrimary: boolean;
}

const ELECTRODES: ElectrodeInfo[] = [
  {
    id: 'C3',
    label: 'C3',
    region: 'Left Motor Strip',
    hemisphere: 'left',
    role: 'YES classification signal',
    detail:
      'Overlies right hemisphere primary motor cortex. Contralateral left-hand motor imagery suppresses mu (10 Hz) rhythm here — this desynchronisation is the primary positive-class feature.',
    isPrimary: true,
  },
  {
    id: 'C4',
    label: 'C4',
    region: 'Right Motor Strip',
    hemisphere: 'right',
    role: 'NO classification signal',
    detail:
      'Overlies left hemisphere primary motor cortex. Contralateral right-hand motor imagery suppresses beta (20 Hz) rhythm here — this provides the negative-class discriminant.',
    isPrimary: true,
  },
  {
    id: 'Cz',
    label: 'Cz',
    region: 'Vertex — Midline',
    hemisphere: 'midline',
    role: 'CSP reference node',
    detail:
      'Midline vertex electrode. Serves as the bilateral balance reference for CSP spatial filter decomposition. Artefacts here will degrade both class boundaries symmetrically.',
    isPrimary: true,
  },
  {
    id: 'FC3',
    label: 'FC3',
    region: 'Left Frontocentral',
    hemisphere: 'left',
    role: 'Supplementary motor',
    detail: 'Premotor / supplementary motor area. Confirms early preparatory activation in the YES component.',
    isPrimary: false,
  },
  {
    id: 'FC4',
    label: 'FC4',
    region: 'Right Frontocentral',
    hemisphere: 'right',
    role: 'Supplementary motor',
    detail: 'Premotor / supplementary motor area. Confirms early preparatory activation in the NO component.',
    isPrimary: false,
  },
  {
    id: 'CP3',
    label: 'CP3',
    region: 'Left Centroparietal',
    hemisphere: 'left',
    role: 'Somatosensory feedback',
    detail: 'Post-central somatosensory strip. Contributes to posterior CSP components and validates spatial filter geometry.',
    isPrimary: false,
  },
  {
    id: 'CP4',
    label: 'CP4',
    region: 'Right Centroparietal',
    hemisphere: 'right',
    role: 'Somatosensory feedback',
    detail: 'Post-central somatosensory strip. Symmetric counterpart to CP3.',
    isPrimary: false,
  },
  {
    id: 'Pz',
    label: 'Pz',
    region: 'Parietal — Midline',
    hemisphere: 'midline',
    role: 'Global reference check',
    detail: 'Posterior parietal. Used as a global reference electrode to monitor broad signal quality and confirm that signal differences are spatially specific.',
    isPrimary: false,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Miniature head-map SVG (simplified 10-20 top view)
// ─────────────────────────────────────────────────────────────────────────────

function HeadMap() {
  const nodes: { id: string; x: number; y: number; primary: boolean }[] = [
    { id: 'C3',  x: 35,  y: 50, primary: true  },
    { id: 'C4',  x: 65,  y: 50, primary: true  },
    { id: 'Cz',  x: 50,  y: 50, primary: true  },
    { id: 'FC3', x: 35,  y: 34, primary: false },
    { id: 'FC4', x: 65,  y: 34, primary: false },
    { id: 'CP3', x: 35,  y: 66, primary: false },
    { id: 'CP4', x: 65,  y: 66, primary: false },
    { id: 'Pz',  x: 50,  y: 70, primary: false },
  ];

  return (
    <svg viewBox="0 0 100 100" className="w-full max-w-[180px]" aria-label="Electrode placement map">
      {/* Skull outline */}
      <ellipse cx="50" cy="50" rx="44" ry="46" fill="none" stroke="#27272a" strokeWidth="1.5" />
      {/* Nose indicator */}
      <path d="M 44 6 Q 50 1 56 6" fill="none" stroke="#27272a" strokeWidth="1" strokeLinecap="round" />
      {/* Ear stubs */}
      <path d="M 6 50 Q 3 46 3 50 Q 3 54 6 54" fill="none" stroke="#27272a" strokeWidth="1" strokeLinecap="round" />
      <path d="M 94 50 Q 97 46 97 50 Q 97 54 94 54" fill="none" stroke="#27272a" strokeWidth="1" strokeLinecap="round" />
      {/* Cross-hairs */}
      <line x1="50" y1="7" x2="50" y2="93" stroke="#1e1e21" strokeWidth="0.5" />
      <line x1="7" y1="50" x2="93" y2="50" stroke="#1e1e21" strokeWidth="0.5" />
      {/* Electrode nodes */}
      {nodes.map(n => (
        <g key={n.id}>
          <circle
            cx={n.x}
            cy={n.y}
            r={n.primary ? 5 : 3.5}
            fill={n.primary ? '#1d4ed8' : '#1e1e21'}
            stroke={n.primary ? '#3b82f6' : '#3f3f46'}
            strokeWidth="1"
          />
          {n.primary && (
            <circle cx={n.x} cy={n.y} r={2} fill="#60a5fa" />
          )}
          <text
            x={n.x}
            y={n.y - 7}
            textAnchor="middle"
            fill={n.primary ? '#93c5fd' : '#52525b'}
            fontSize="5.5"
            fontFamily="'JetBrains Mono', monospace"
            fontWeight={n.primary ? '700' : '400'}
          >
            {n.id}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Single electrode card
// ─────────────────────────────────────────────────────────────────────────────

function ElectrodeCard({ e }: { e: ElectrodeInfo }) {
  const hemiColor =
    e.hemisphere === 'left'    ? 'text-blue-400 border-blue-900 bg-blue-950/40'
    : e.hemisphere === 'right' ? 'text-violet-400 border-violet-900 bg-violet-950/40'
    : 'text-zinc-300 border-zinc-700 bg-zinc-800/40';

  return (
    <div className={cn(
      'rounded-lg border p-3 space-y-1',
      e.isPrimary
        ? 'border-zinc-700 bg-zinc-900/70'
        : 'border-zinc-800/60 bg-zinc-900/30',
    )}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {e.isPrimary
            ? <Radio className="h-3.5 w-3.5 text-blue-500 shrink-0" />
            : <Circle className="h-3 w-3 text-zinc-600 shrink-0" />
          }
          <span className={cn(
            'font-mono text-sm font-bold',
            e.isPrimary ? 'text-slate-100' : 'text-zinc-400',
          )}>
            {e.label}
          </span>
          <span className="text-xs text-zinc-500">{e.region}</span>
        </div>
        <span className={cn(
          'rounded border px-1.5 py-0.5 text-[9px] font-semibold tracking-wide uppercase',
          hemiColor,
        )}>
          {e.hemisphere}
        </span>
      </div>
      <p className="text-[10px] font-medium text-zinc-400">{e.role}</p>
      <p className="text-[9px] leading-relaxed text-zinc-600">{e.detail}</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PlacementGuide (public export)
// ─────────────────────────────────────────────────────────────────────────────

interface PlacementGuideProps {
  open: boolean;
  onToggle: () => void;
}

export default function PlacementGuide({ open, onToggle }: PlacementGuideProps) {
  const primaryNodes  = ELECTRODES.filter(e => e.isPrimary);
  const secondaryNodes = ELECTRODES.filter(e => !e.isPrimary);

  return (
    <div
      className={cn(
        'relative flex shrink-0 flex-col transition-all duration-300 ease-in-out',
        open ? 'w-[300px]' : 'w-9',
      )}
    >
      {/* Toggle strip */}
      <button
        onClick={onToggle}
        title={open ? 'Collapse placement guide' : 'Open placement guide'}
        className={cn(
          'absolute -left-3 top-4 z-10 flex h-7 w-7 items-center justify-center',
          'rounded-full border border-zinc-700 bg-zinc-900 shadow-lg',
          'text-zinc-400 transition-colors hover:border-blue-600 hover:text-blue-400',
        )}
      >
        {open
          ? <ChevronRight className="h-3.5 w-3.5" />
          : <ChevronLeft className="h-3.5 w-3.5" />
        }
      </button>

      {/* Panel content — hidden when collapsed */}
      <div className={cn(
        'flex h-full flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/60',
        'transition-opacity duration-200',
        open ? 'opacity-100' : 'opacity-0 pointer-events-none',
      )}>
        {/* Panel header */}
        <div className="border-b border-zinc-800 px-4 py-3">
          <h2 className="text-xs font-bold tracking-widest text-slate-200 uppercase">
            Sensorimotor Placement Guide
          </h2>
          <p className="mt-0.5 text-[10px] text-zinc-600">10-20 international system</p>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
          {/* Mini head map */}
          <div className="flex justify-center py-1">
            <HeadMap />
          </div>

          {/* Legend */}
          <div className="flex items-center gap-3 text-[9px] text-zinc-600">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-blue-600 ring-1 ring-blue-400" />
              Primary node
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-zinc-700 ring-1 ring-zinc-600" />
              Secondary node
            </span>
          </div>

          {/* Primary nodes */}
          <div>
            <div className="mb-2 text-[9px] font-bold tracking-widest text-zinc-500 uppercase">
              Primary — Classification Nodes
            </div>
            <div className="space-y-2">
              {primaryNodes.map(e => <ElectrodeCard key={e.id} e={e} />)}
            </div>
          </div>

          {/* Secondary nodes */}
          <div>
            <div className="mb-2 text-[9px] font-bold tracking-widest text-zinc-500 uppercase">
              Secondary — Spatial Filter Nodes
            </div>
            <div className="space-y-2">
              {secondaryNodes.map(e => <ElectrodeCard key={e.id} e={e} />)}
            </div>
          </div>

          {/* Impedance warning */}
          <div className="rounded-lg border border-amber-800/50 bg-amber-950/30 p-3">
            <div className="mb-1.5 flex items-center gap-2">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
              <span className="text-[10px] font-bold tracking-wider text-amber-400 uppercase">
                Impedance Check
              </span>
            </div>
            <p className="text-[9px] leading-relaxed text-amber-700">
              Verify <strong className="text-amber-500">all</strong> primary node impedances are{' '}
              <strong className="text-amber-400">&lt; 10 kΩ</strong> before starting acquisition.
              Elevated impedance at <strong className="text-amber-500">C3 / C4</strong> will
              degrade the CSP spatial filter and inflate the misclassification rate. Scalp prep
              with abrasive gel is recommended.
            </p>
          </div>

          {/* C3/C4 classification boundary note */}
          <div className="rounded-lg border border-blue-900/40 bg-blue-950/20 p-3">
            <p className="text-[9px] leading-relaxed text-blue-700">
              <strong className="text-blue-400">C3 ↔ C4 power asymmetry</strong> is the primary
              discriminant boundary learned by the CSP decomposition. The classifier maps
              C3-dominant epochs → <span className="font-bold text-green-400">YES</span> and
              C4-dominant epochs → <span className="font-bold text-red-400">NO</span>.
            </p>
          </div>
        </div>
      </div>

      {/* Collapsed label (rotated) */}
      {!open && (
        <div className="flex flex-1 items-center justify-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40">
          <span
            className="whitespace-nowrap font-semibold text-[10px] tracking-widest text-zinc-600 uppercase"
            style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
          >
            Placement Guide
          </span>
        </div>
      )}
    </div>
  );
}
