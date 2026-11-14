import React, { useMemo, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import {
  SCAN_PRESETS,
  ScanPreset,
  ScanStepInput,
  simulateMambaScan,
} from '../lib/mambaSim';

export const ScanLabSection: React.FC = () => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>(SCAN_PRESETS[0].id);
  const [paramA, setParamA] = useState<number>(SCAN_PRESETS[0].A);
  const [inputs, setInputs] = useState<ScanStepInput[]>(SCAN_PRESETS[0].inputs);
  const [selectedNode, setSelectedNode] = useState<{ round: number; t: number } | null>(null);

  const activePreset = useMemo<ScanPreset | undefined>(
    () => SCAN_PRESETS.find((p) => p.id === selectedPresetId),
    [selectedPresetId],
  );

  const sim = useMemo(() => simulateMambaScan(paramA, inputs), [paramA, inputs]);

  const handleSelectPreset = (preset: ScanPreset) => {
    setSelectedPresetId(preset.id);
    setParamA(preset.A);
    setInputs(preset.inputs.map((s) => ({ ...s })));
    setSelectedNode(null);
  };

  const handleStepChange = (idx: number, field: 'x' | 'delta', value: number) => {
    setInputs((prev) =>
      prev.map((item, i) => (i === idx ? { ...item, [field]: value } : item)),
    );
  };

  const handleReset = () => {
    if (activePreset) {
      handleSelectPreset(activePreset);
    }
  };

  const isEquivalent = sim.maxAbsDiff <= 1e-5;
  const inspectedNode =
    selectedNode &&
    sim.rounds[selectedNode.round] &&
    sim.rounds[selectedNode.round][selectedNode.t]
      ? sim.rounds[selectedNode.round][selectedNode.t]
      : null;

  const inspectedEarlier =
    selectedNode &&
    inspectedNode &&
    inspectedNode.sourceEarlierT !== null &&
    sim.rounds[selectedNode.round - 1]
      ? sim.rounds[selectedNode.round - 1][inspectedNode.sourceEarlierT]
      : null;

  const inspectedLater =
    selectedNode &&
    inspectedNode &&
    inspectedNode.sourceEarlierT !== null &&
    sim.rounds[selectedNode.round - 1]
      ? sim.rounds[selectedNode.round - 1][selectedNode.t]
      : null;

  const svgWidth = Math.max(560, inputs.length * 92 + 64);
  const rowHeight = 68;
  const svgHeight = sim.rounds.length * rowHeight + 44;

  return (
    <section id="scan-lab" aria-labelledby="scan-lab-heading" className="scroll-mt-20">
      <div className="mb-6 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs font-medium text-slate-500">
            Phase 1–3 · Zero-Order Hold &amp; Associative Prefix Recurrence
          </p>
          <h1
            id="scan-lab-heading"
            className="mt-1 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl"
            style={{ textWrap: 'balance' }}
          >
            01. Selective Discretization &amp; Parallel Scan Lab
          </h1>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono text-slate-600 tabular-nums">
          <span>
            Max Diff:{' '}
            <strong className="text-slate-900">{sim.maxAbsDiff.toExponential(2)}</strong>
          </span>
          <span aria-hidden="true">/</span>
          <span className={isEquivalent ? 'font-semibold text-emerald-700' : 'font-semibold text-amber-700'}>
            {isEquivalent ? '● MATCH (<= 1e-5)' : '▲ DRIFT'}
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-6 lg:flex-row lg:items-stretch">
        {/* Left Control Column */}
        <div className="flex w-full flex-col justify-between rounded-lg border border-slate-200 bg-white p-5 lg:w-[360px] lg:shrink-0">
          <div className="space-y-5">
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Verification Presets
                </span>
                <button
                  type="button"
                  onClick={handleReset}
                  className="inline-flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-900 whitespace-nowrap shrink-0"
                >
                  <RotateCcw className="h-3 w-3" aria-hidden="true" />
                  <span>Reset</span>
                </button>
              </div>
              <div className="grid grid-cols-1 gap-1.5" role="group" aria-label="Scan presets">
                {SCAN_PRESETS.map((preset) => {
                  const active = preset.id === selectedPresetId;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => handleSelectPreset(preset)}
                      className={`w-full rounded-md border px-3 py-2 text-left text-xs font-medium transition-colors whitespace-nowrap truncate ${
                        active
                          ? 'border-blue-600 bg-blue-50 text-blue-800'
                          : 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
              {activePreset && (
                <p className="mt-2.5 text-xs leading-relaxed text-slate-600">
                  {activePreset.description}
                </p>
              )}
            </div>

            <div className="border-t border-slate-200 pt-4">
              <div className="mb-1.5 flex items-center justify-between">
                <label htmlFor="param-a-slider" className="text-xs font-semibold text-slate-700">
                  Continuous Diagonal A
                </label>
                <span className="font-mono text-xs font-semibold text-slate-900 tabular-nums">
                  {paramA.toFixed(2)}
                </span>
              </div>
              <input
                id="param-a-slider"
                type="range"
                min={-4}
                max={-0.1}
                step={0.05}
                value={paramA}
                onChange={(e) => setParamA(parseFloat(e.target.value))}
                className="w-full accent-blue-600"
              />
              <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400">
                <span>-4.00 (fast decay)</span>
                <span>-0.10 (slow decay)</span>
              </div>
            </div>

            <div className="border-t border-slate-200 pt-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-700">
                  Per-Token Inputs &amp; Step Size
                </span>
                <span className="font-mono text-[11px] text-slate-500">
                  {inputs.length} steps
                </span>
              </div>
              <div className="max-h-60 space-y-2 overflow-y-auto pr-1">
                {inputs.map((step, idx) => (
                  <div
                    key={step.t}
                    className="grid grid-cols-[36px_1fr_1fr] items-center gap-2 rounded border border-slate-200 bg-slate-50 px-2.5 py-1.5"
                  >
                    <span className="font-mono text-xs font-semibold text-slate-600">
                      t={step.t}
                    </span>
                    <label className="flex items-center justify-between gap-1 text-[11px] text-slate-600">
                      <span className="font-mono">x</span>
                      <input
                        type="number"
                        step={0.5}
                        aria-label={`x at step ${step.t}`}
                        value={Number(step.x.toFixed(3))}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          if (!Number.isNaN(v)) handleStepChange(idx, 'x', v);
                        }}
                        className="w-16 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-right font-mono text-xs text-slate-900 tabular-nums"
                      />
                    </label>
                    <label className="flex items-center justify-between gap-1 text-[11px] text-slate-600">
                      <span className="font-mono">Δ</span>
                      <input
                        type="number"
                        min={0.01}
                        max={6}
                        step={0.05}
                        aria-label={`Delta at step ${step.t}`}
                        value={Number(step.delta.toFixed(3))}
                        onChange={(e) => {
                          const v = Math.max(0.001, parseFloat(e.target.value) || 0.01);
                          handleStepChange(idx, 'delta', v);
                        }}
                        className="w-16 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-right font-mono text-xs text-slate-900 tabular-nums"
                      />
                    </label>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-5 border-t border-slate-200 pt-4 text-xs">
            <div className="flex items-center justify-between text-slate-600">
              <span>3-Pair Associativity Error</span>
              <span className="font-mono font-semibold text-slate-900 tabular-nums">
                {sim.associativityMaxErr.toExponential(2)}
              </span>
            </div>
            <div className="mt-1 flex items-center justify-between text-slate-600">
              <span>ZOH Invariant</span>
              <span className="font-mono text-emerald-700">
                ● A-bar in (0, 1)
              </span>
            </div>
          </div>
        </div>

        {/* Right Stage: Discretization Table + Parallel Prefix Tree */}
        <div className="flex min-w-0 flex-1 flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-slate-900">
                ZOH Discretization &amp; Sequential vs. Parallel State Trace
              </h2>
              <span className="font-mono text-xs text-slate-500">
                A-bar = exp(Δ·A) · B-bar = ((exp(Δ·A)-1)/A)·B
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left font-mono text-xs tabular-nums">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-3 font-medium">t</th>
                    <th className="py-2 px-3 font-medium">x_t</th>
                    <th className="py-2 px-3 font-medium">Delta_t</th>
                    <th className="py-2 px-3 font-medium">A-bar</th>
                    <th className="py-2 px-3 font-medium">B-bar</th>
                    <th className="py-2 px-3 font-medium text-blue-700">h_seq</th>
                    <th className="py-2 px-3 font-medium text-emerald-700">h_par</th>
                    <th className="py-2 pl-3 font-medium">y_t</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {sim.steps.map((row) => (
                    <tr key={row.t} className="hover:bg-slate-50">
                      <td className="py-2 pr-3 font-semibold text-slate-700">{row.t}</td>
                      <td className="py-2 px-3 text-slate-800">{row.x.toFixed(3)}</td>
                      <td className="py-2 px-3 text-slate-800">{row.delta.toFixed(3)}</td>
                      <td className="py-2 px-3 text-slate-700">{row.aBar.toFixed(4)}</td>
                      <td className="py-2 px-3 text-slate-700">{row.bBar.toFixed(4)}</td>
                      <td className="py-2 px-3 font-medium text-blue-700">{row.hSeq.toFixed(4)}</td>
                      <td className="py-2 px-3 font-medium text-emerald-700">{row.hPar.toFixed(4)}</td>
                      <td className="py-2 pl-3 font-semibold text-slate-900">{row.yPar.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold text-slate-900">
                  Hillis-Steele Parallel Associative Scan Tree
                </h2>
                <p className="text-xs text-slate-500">
                  Click any node to inspect compose((a_later, b_later), (a_earlier, b_earlier)).
                </p>
              </div>
              <span className="font-mono text-xs text-slate-500">
                {sim.rounds.length - 1} parallel stride rounds
              </span>
            </div>

            <div className="overflow-x-auto rounded border border-slate-100 bg-slate-50 p-3">
              <svg
                width={svgWidth}
                height={svgHeight}
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                role="img"
                aria-label="Hillis-Steele parallel associative scan binary tree"
                className="mx-auto block"
              >
                {sim.rounds.map((roundNodes, rIdx) => {
                  const y = 24 + rIdx * rowHeight;
                  return (
                    <g key={rIdx}>
                      <text
                        x={8}
                        y={y + 20}
                        className="fill-slate-400 font-mono text-[10px]"
                      >
                        {rIdx === 0 ? 'Init' : `S=${1 << (rIdx - 1)}`}
                      </text>
                      {roundNodes.map((node) => {
                        const x = 68 + node.t * 92;
                        const isSelected =
                          selectedNode?.round === rIdx && selectedNode?.t === node.t;
                        const hasMerge = node.sourceEarlierT !== null;
                        const prevY = 24 + (rIdx - 1) * rowHeight + 36;
                        const earlierX =
                          node.sourceEarlierT !== null ? 68 + node.sourceEarlierT * 92 + 36 : 0;

                        return (
                          <g key={`${rIdx}-${node.t}`}>
                            {rIdx > 0 && (
                              <line
                                x1={x + 36}
                                y1={prevY}
                                x2={x + 36}
                                y2={y}
                                stroke="#CBD5E1"
                                strokeWidth={1.25}
                              />
                            )}
                            {hasMerge && (
                              <line
                                x1={earlierX}
                                y1={prevY}
                                x2={x + 36}
                                y2={y}
                                stroke="#2563EB"
                                strokeWidth={1.5}
                                strokeDasharray="3 2"
                              />
                            )}
                            <g
                              role="button"
                              tabIndex={0}
                              aria-label={`Round ${rIdx} step ${node.t} a=${node.a.toFixed(3)} b=${node.b.toFixed(3)}`}
                              onClick={() => setSelectedNode({ round: rIdx, t: node.t })}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                  e.preventDefault();
                                  setSelectedNode({ round: rIdx, t: node.t });
                                }
                              }}
                              className="cursor-pointer"
                            >
                              <rect
                                x={x}
                                y={y}
                                width={76}
                                height={36}
                                rx={6}
                                fill={isSelected ? '#EFF6FF' : '#FFFFFF'}
                                stroke={isSelected ? '#2563EB' : hasMerge ? '#93C5FD' : '#CBD5E1'}
                                strokeWidth={isSelected ? 2 : 1}
                              />
                              <text
                                x={x + 38}
                                y={y + 14}
                                textAnchor="middle"
                                className="fill-slate-500 font-mono text-[10px]"
                              >
                                a:{node.a.toFixed(2)}
                              </text>
                              <text
                                x={x + 38}
                                y={y + 28}
                                textAnchor="middle"
                                className="fill-slate-900 font-mono text-[11px] font-semibold"
                              >
                                b:{node.b.toFixed(3)}
                              </text>
                            </g>
                          </g>
                        );
                      })}
                    </g>
                  );
                })}
              </svg>
            </div>

            {inspectedNode && (
              <div className="mt-3 rounded border border-slate-200 bg-slate-50 px-3.5 py-2.5 font-mono text-xs text-slate-700">
                {inspectedEarlier && inspectedLater ? (
                  <span>
                    compose(later=({inspectedLater.a.toFixed(4)}, {inspectedLater.b.toFixed(4)}),
                    earlier=({inspectedEarlier.a.toFixed(4)}, {inspectedEarlier.b.toFixed(4)})) = (
                    <strong className="text-blue-700">{inspectedNode.a.toFixed(4)}</strong>,{' '}
                    <strong className="text-emerald-700">{inspectedNode.b.toFixed(4)}</strong>)
                  </span>
                ) : (
                  <span>
                    Pass-through node at round {selectedNode?.round}, t={inspectedNode.t}: (a=
                    {inspectedNode.a.toFixed(4)}, b={inspectedNode.b.toFixed(4)})
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
