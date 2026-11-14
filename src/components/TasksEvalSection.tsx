import React, { useMemo, useRef, useState } from 'react';
import { RefreshCw, Upload } from 'lucide-react';
import { generateInductionPreview, generateSelectiveCopyPreview } from '../lib/dataGenerators';
import { TrainEvalReport } from '../types/reports';

interface TasksEvalSectionProps {
  trainEval: TrainEvalReport | null;
  onUploadTrainEval?: (report: TrainEvalReport) => void;
}

export const TasksEvalSection: React.FC<TasksEvalSectionProps> = ({ trainEval, onUploadTrainEval }) => {
  const [activeTask, setActiveTask] = useState<'selective_copy' | 'induction'>('selective_copy');
  const [previewLength, setPreviewLength] = useState<64 | 256>(64);
  const [seed, setSeed] = useState<number>(20261114);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !onUploadTrainEval) return;
    try {
      const parsed = JSON.parse(await file.text()) as TrainEvalReport;
      if (!parsed.tasks?.selective_copy || !parsed.tasks?.induction) {
        throw new Error('JSON must contain tasks.selective_copy and tasks.induction.');
      }
      setUploadErr(null);
      onUploadTrainEval(parsed);
    } catch (err) {
      setUploadErr(err instanceof Error ? err.message : 'Invalid train_eval.json file.');
    }
  };

  const seq = useMemo(() => {
    if (activeTask === 'selective_copy') {
      return generateSelectiveCopyPreview(previewLength, seed, previewLength === 64 ? 8 : 16);
    }
    return generateInductionPreview(previewLength, seed);
  }, [activeTask, previewLength, seed]);

  const currentTaskReport = trainEval ? trainEval.tasks[activeTask] : null;
  const evalLengths = trainEval?.eval_lengths ?? [256, 1024, 4096];

  return (
    <section id="tasks-eval" aria-labelledby="tasks-eval-heading" className="scroll-mt-20 border-t border-slate-200 pt-12">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-medium text-slate-500">
            Phase 5–6 · Trained at Length 256 · Zero-Shot Extrapolation to 1024 and 4096
          </p>
          <h2 id="tasks-eval-heading" className="mt-1 text-2xl font-bold tracking-tight text-slate-900">
            02. Synthetic Memory Tasks &amp; Length Generalization
          </h2>
        </div>

        <div className="inline-flex items-center gap-1 rounded-lg bg-slate-200/70 p-1" role="tablist" aria-label="Synthetic benchmark task">
          <button
            type="button"
            role="tab"
            aria-selected={activeTask === 'selective_copy'}
            onClick={() => setActiveTask('selective_copy')}
            className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              activeTask === 'selective_copy'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Selective Copying
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTask === 'induction'}
            onClick={() => setActiveTask('induction')}
            className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              activeTask === 'induction'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Induction Heads
          </button>
        </div>
      </div>

      {/* Empirical Length-Generalization Results & Parameter Parity */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Accuracy Chart */}
        <div className="rounded-lg border border-slate-200 bg-white p-5 lg:col-span-7">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Masked Token Accuracy vs. Evaluation Length
              </h3>
              <p className="text-xs text-slate-500">
                Both models: 2 layers, d_model=64, AdamW lr=1e-3, batch=32, seed=20261114
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="inline-flex items-center gap-1.5 font-medium text-blue-700">
                <span className="inline-block h-2.5 w-2.5 rounded-xs bg-blue-600" aria-hidden="true" />
                MambaLM
              </span>
              <span className="inline-flex items-center gap-1.5 font-medium text-amber-700">
                <span className="inline-block h-2.5 w-2.5 rounded-xs bg-amber-600" aria-hidden="true" />
                Causal Transformer
              </span>
            </div>
          </div>

          {currentTaskReport ? (
            <div className="space-y-4">
              <svg
                viewBox="0 0 600 240"
                className="w-full"
                role="img"
                aria-label={`Length generalization chart for ${activeTask}`}
              >
                {/* Horizontal gridlines */}
                {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => {
                  const y = 200 - tick * 170;
                  return (
                    <g key={tick}>
                      <line x1={52} y1={y} x2={570} y2={y} stroke="#E2E8F0" strokeWidth={1} />
                      <text x={44} y={y + 4} textAnchor="end" className="fill-slate-400 font-mono text-[11px]">
                        {(tick * 100).toFixed(0)}%
                      </text>
                    </g>
                  );
                })}
                {/* Pass floor line */}
                {(() => {
                  const floorVal = activeTask === 'selective_copy' ? 0.8 : 0.5;
                  const yFloor = 200 - floorVal * 170;
                  return (
                    <g>
                      <line
                        x1={52}
                        y1={yFloor}
                        x2={570}
                        y2={yFloor}
                        stroke="#059669"
                        strokeWidth={1.25}
                        strokeDasharray="4 4"
                      />
                      <text x={568} y={yFloor - 5} textAnchor="end" className="fill-emerald-700 font-mono text-[10px]">
                        L=256 floor ({(floorVal * 100).toFixed(0)}%)
                      </text>
                    </g>
                  );
                })()}

                {/* Lines & points */}
                {(() => {
                  const xs = [110, 310, 510];
                  const mambaPts = evalLengths.map((len, i) => {
                    const acc = currentTaskReport.models.mamba.lengths[String(len)]?.token_accuracy ?? 0;
                    return { x: xs[i], y: 200 - acc * 170, acc, len };
                  });
                  const tfPts = evalLengths.map((len, i) => {
                    const acc = currentTaskReport.models.transformer.lengths[String(len)]?.token_accuracy ?? 0;
                    return { x: xs[i], y: 200 - acc * 170, acc, len };
                  });

                  const toPath = (pts: { x: number; y: number }[]) =>
                    pts.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

                  return (
                    <g>
                      <path d={toPath(mambaPts)} fill="none" stroke="#2563EB" strokeWidth={2.5} />
                      <path d={toPath(tfPts)} fill="none" stroke="#D97706" strokeWidth={2.5} strokeDasharray="6 3" />

                      {mambaPts.map((p, idx) => {
                        const tfY = tfPts[idx].y;
                        const isMambaHigher = p.y <= tfY;
                        const labelY = isMambaHigher ? Math.max(18, p.y - 10) : Math.min(212, p.y + 16);
                        return (
                          <g key={`m-${p.len}`}>
                            <circle cx={p.x} cy={p.y} r={5} fill="#2563EB" stroke="#FFFFFF" strokeWidth={1.5} />
                            <text
                              x={p.x}
                              y={labelY}
                              textAnchor="middle"
                              className="fill-blue-700 font-mono text-[11px] font-semibold"
                            >
                              {(p.acc * 100).toFixed(1)}%
                            </text>
                          </g>
                        );
                      })}
                      {tfPts.map((p, idx) => {
                        const mY = mambaPts[idx].y;
                        const isTfHigher = p.y < mY;
                        const labelY = isTfHigher ? Math.max(18, p.y - 10) : Math.min(212, p.y + 18);
                        return (
                          <g key={`t-${p.len}`}>
                            <rect
                              x={p.x - 4.5}
                              y={p.y - 4.5}
                              width={9}
                              height={9}
                              fill="#D97706"
                              stroke="#FFFFFF"
                              strokeWidth={1.5}
                            />
                            <text
                              x={p.x}
                              y={labelY}
                              textAnchor="middle"
                              className="fill-amber-700 font-mono text-[11px] font-semibold"
                            >
                              {(p.acc * 100).toFixed(1)}%
                            </text>
                          </g>
                        );
                      })}

                      {xs.map((x, idx) => (
                        <text
                          key={evalLengths[idx]}
                          x={x}
                          y={228}
                          textAnchor="middle"
                          className="fill-slate-600 font-mono text-xs font-medium"
                        >
                          L={evalLengths[idx]}
                        </text>
                      ))}
                    </g>
                  );
                })()}
              </svg>
            </div>
          ) : (
            <div className="flex h-56 flex-col items-center justify-center gap-3 rounded border border-dashed border-slate-200 bg-slate-50 px-6 text-center">
              <p className="text-xs font-medium text-slate-700">
                Awaiting GPU training &amp; evaluation results (<code className="font-mono">reports/train_eval.json</code>)
              </p>
              <p className="max-w-md text-xs text-slate-500">
                Run <code className="font-mono">python -m src.train</code> and <code className="font-mono">python -m src.eval</code> on GPU, then load your generated <code className="font-mono">train_eval.json</code> here.
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/json,.json"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100 whitespace-nowrap shrink-0"
              >
                <Upload className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Load train_eval.json</span>
              </button>
              {uploadErr && <span className="text-xs text-rose-600">{uploadErr}</span>}
            </div>
          )}
        </div>

        {/* Parameter Parity & Exact Numeric Table */}
        <div className="flex flex-col justify-between rounded-lg border border-slate-200 bg-white p-5 lg:col-span-5">
          {currentTaskReport ? (
            <div className="space-y-5">
              <div>
                <h3 className="text-sm font-semibold text-slate-900">
                  Parameter Parity &amp; Checkpoint Readout
                </h3>
                <p className="mt-0.5 text-xs text-slate-500">
                  Transformer MLP width tuned at 2 layers so trainable parameter counts stay within 10%.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-3 border-y border-slate-200 py-3">
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                    Mamba Params
                  </div>
                  <div className="mt-1 font-mono text-lg font-semibold text-slate-900 tabular-nums">
                    {currentTaskReport.mamba_param_count.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                    TF Params
                  </div>
                  <div className="mt-1 font-mono text-lg font-semibold text-slate-900 tabular-nums">
                    {currentTaskReport.transformer_param_count.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                    MLP Width
                  </div>
                  <div className="mt-1 font-mono text-lg font-semibold text-slate-900 tabular-nums">
                    {currentTaskReport.transformer_mlp_width}
                  </div>
                  <div className="font-mono text-[11px] text-emerald-700">
                    {(currentTaskReport.param_relative_diff * 100).toFixed(2)}% delta
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left font-mono text-xs tabular-nums">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 pr-2 font-medium">Model</th>
                      <th className="py-2 px-2 font-medium">Steps</th>
                      <th className="py-2 px-2 font-medium">L=256</th>
                      <th className="py-2 px-2 font-medium">L=1024</th>
                      <th className="py-2 pl-2 font-medium">L=4096</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    <tr>
                      <td className="py-2.5 pr-2 font-semibold text-blue-700">MambaLM</td>
                      <td className="py-2.5 px-2 text-slate-600">{currentTaskReport.mamba_step_count}</td>
                      {evalLengths.map((len) => (
                        <td key={len} className="py-2.5 px-2 font-semibold text-slate-900">
                          {(currentTaskReport.models.mamba.lengths[String(len)]?.token_accuracy ?? 0).toFixed(3)}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="py-2.5 pr-2 font-semibold text-amber-700">Transformer</td>
                      <td className="py-2.5 px-2 text-slate-600">{currentTaskReport.transformer_step_count}</td>
                      {evalLengths.map((len) => (
                        <td key={len} className="py-2.5 px-2 font-semibold text-slate-900">
                          {(currentTaskReport.models.transformer.lengths[String(len)]?.token_accuracy ?? 0).toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>

              {(() => {
                const mamba256 = currentTaskReport.models.mamba.lengths['256']?.token_accuracy ?? 0;
                const floor = activeTask === 'selective_copy' ? 0.8 : 0.5;
                const passed = mamba256 >= floor;
                return (
                  <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                    <strong className={passed ? 'text-emerald-700' : 'text-rose-700'}>
                      {passed ? '● FLOOR MET:' : '▲ BELOW FLOOR:'}
                    </strong>{' '}
                    MambaLM {passed ? 'passes' : 'scored below'} L=256 ({(mamba256 * 100).toFixed(1)}%{' '}
                    {passed ? '≥' : '<'} {(floor * 100).toFixed(1)}%) on 64 fresh sequences.
                  </div>
                );
              })()}
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-900">
                  Verified Architecture Parity (Pre-Training)
                </h3>
                <p className="mt-0.5 text-xs text-slate-500">
                  Exact parameter counts measured from constructed PyTorch modules before training.
                </p>
              </div>
              <div className="grid grid-cols-3 gap-3 border-y border-slate-200 py-3 font-mono tabular-nums">
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                    Mamba Params
                  </div>
                  <div className="mt-1 text-lg font-semibold text-slate-900">
                    {activeTask === 'selective_copy' ? '98,432' : '98,176'}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                    TF Params
                  </div>
                  <div className="mt-1 text-lg font-semibold text-slate-900">
                    {activeTask === 'selective_copy' ? '98,272' : '98,016'}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                    MLP Width
                  </div>
                  <div className="mt-1 text-lg font-semibold text-slate-900">240</div>
                  <div className="text-[11px] text-emerald-700">0.16% delta</div>
                </div>
              </div>
              <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                Target L=256 floors: <strong className="font-mono text-slate-900">&ge; 0.80</strong> on Selective Copying and <strong className="font-mono text-slate-900">&ge; 0.50</strong> on Induction Heads.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Interactive Sequence Generator Explorer */}
      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Deterministic Task Generator &amp; Loss Mask Inspector
            </h3>
            <p className="text-xs text-slate-500">
              {activeTask === 'selective_copy'
                ? 'Id 0 = noise, 1–16 = content tokens (blue), 17 = recall marker (emerald). Loss mask = 1 strictly on last 16 positions.'
                : 'Vocab 0–15. Four key-value bigrams in the first half (blue/amber), final token queries one key (emerald). Loss mask = 1 only at last index.'}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="inline-flex items-center gap-1 rounded-md bg-slate-100 p-1">
              {([64, 256] as const).map((len) => (
                <button
                  key={len}
                  type="button"
                  onClick={() => setPreviewLength(len)}
                  className={`rounded px-2.5 py-1 font-mono text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    previewLength === len ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                  }`}
                >
                  L={len}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => setSeed((s) => s + 1)}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 font-mono text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
            >
              <RefreshCw className="h-3 w-3" aria-hidden="true" />
              <span>Seed {seed}</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-8 gap-1 font-mono text-[11px] sm:grid-cols-16">
          {seq.inputIds.map((tok, idx) => {
            const isMasked = seq.lossMask[idx] === 1;
            const isContent =
              activeTask === 'selective_copy'
                ? tok >= 1 && tok <= 16 && !isMasked
                : seq.bigramStarts?.includes(idx);
            const isValue =
              activeTask === 'induction' && seq.bigramStarts?.some((s) => s + 1 === idx);

            let cellCls = 'border-slate-200 bg-slate-50 text-slate-400';
            if (isMasked) {
              cellCls = 'border-emerald-300 bg-emerald-50 font-semibold text-emerald-900';
            } else if (isContent) {
              cellCls = 'border-blue-300 bg-blue-50 font-semibold text-blue-800';
            } else if (isValue) {
              cellCls = 'border-amber-300 bg-amber-50 font-semibold text-amber-800';
            }

            return (
              <div
                key={idx}
                title={`pos=${idx} | in=${tok} | target=${seq.targetIds[idx]} | mask=${seq.lossMask[idx]}`}
                className={`flex flex-col items-center justify-center rounded border py-1 tabular-nums ${cellCls}`}
              >
                <span>{tok}</span>
                {isMasked && (
                  <span className="text-[9px] text-emerald-700">→{seq.targetIds[idx]}</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
