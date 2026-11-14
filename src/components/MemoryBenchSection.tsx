import React, { useState } from 'react';
import { Play, RotateCcw, StepForward } from 'lucide-react';
import { InferenceReport } from '../types/reports';

interface MemoryBenchSectionProps {
  inference: InferenceReport | null;
}

const MAMBA_CONST_BYTES = 2 * (128 * 16 + 128 * 3) * 4; // 19,456 bytes (2 layers, fp32)
const TF_BYTES_PER_TOKEN = 2 * 2 * (4 * 16) * 4; // 1,024 bytes per token (2 layers, K+V, fp32)

export const MemoryBenchSection: React.FC<MemoryBenchSectionProps> = ({ inference }) => {
  const [chartMode, setChartMode] = useState<'interactive' | 'matplotlib'>('interactive');
  const [stepCount, setStepCount] = useState<number>(256);

  const mambaStepBytes = MAMBA_CONST_BYTES;
  const tfStepBytes = stepCount * TF_BYTES_PER_TOKEN;

  const lengths = inference?.lengths ?? [256, 1024, 4096];
  const mambaBytes = inference?.mamba_state_bytes ?? [MAMBA_CONST_BYTES, MAMBA_CONST_BYTES, MAMBA_CONST_BYTES];
  const tfBytes = inference?.transformer_kv_bytes ?? [262144, 1048576, 4194304];
  const mambaMs = inference?.mamba_latency_ms ?? [0, 0, 0];
  const tfMs = inference?.transformer_latency_ms ?? [0, 0, 0];

  const maxKib = Math.max(...tfBytes, ...mambaBytes) / 1024;
  const maxMs = Math.max(...tfMs, ...mambaMs, 1);

  return (
    <section id="memory-bench" aria-labelledby="memory-bench-heading" className="scroll-mt-20 border-t border-slate-200 pt-12">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-medium text-slate-500">
            Phase 7 · Recurrent Inference State vs. Growing Key-Value Cache (batch=1, fp32)
          </p>
          <h2 id="memory-bench-heading" className="mt-1 text-2xl font-bold tracking-tight text-slate-900">
            03. Recurrent Memory &amp; Per-Token Latency Benchmark
          </h2>
        </div>

        <div className="inline-flex items-center gap-1 rounded-lg bg-slate-200/70 p-1" role="tablist" aria-label="Chart renderer">
          <button
            type="button"
            role="tab"
            aria-selected={chartMode === 'interactive'}
            onClick={() => setChartMode('interactive')}
            className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              chartMode === 'interactive' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Interactive Charts
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={chartMode === 'matplotlib'}
            onClick={() => setChartMode('matplotlib')}
            className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              chartMode === 'matplotlib' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Matplotlib PNGs
          </button>
        </div>
      </div>

      {/* KPI Strip */}
      {inference && (
        <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Mamba State (L=256..4096)
            </div>
            <div className="mt-1 font-mono text-2xl font-bold text-slate-900 tabular-nums">
              {(mambaBytes[0] / 1024).toFixed(1)}{' '}
              <span className="text-xs font-normal text-slate-400">KiB</span>
            </div>
            <div className="mt-1 font-mono text-xs text-emerald-700">
              ● Ratio {inference.mamba_state_byte_ratio.toFixed(2)} (&lt;= 1.05)
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Transformer KV (L=4096)
            </div>
            <div className="mt-1 font-mono text-2xl font-bold text-slate-900 tabular-nums">
              {(tfBytes[2] / 1024).toFixed(1)}{' '}
              <span className="text-xs font-normal text-slate-400">KiB</span>
            </div>
            <div className="mt-1 font-mono text-xs text-amber-700">
              ▲ {inference.transformer_4096_over_256_ratio.toFixed(1)}x vs L=256 (&gt;= 8.0x)
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Mamba Step Latency (L=4096)
            </div>
            <div className="mt-1 font-mono text-2xl font-bold text-slate-900 tabular-nums">
              {mambaMs[2].toFixed(3)}{' '}
              <span className="text-xs font-normal text-slate-400">ms/tok</span>
            </div>
            <div className="mt-1 font-mono text-xs text-slate-500">
              Device: {inference.device.toUpperCase()}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Transformer Latency (L=4096)
            </div>
            <div className="mt-1 font-mono text-2xl font-bold text-slate-900 tabular-nums">
              {tfMs[2].toFixed(3)}{' '}
              <span className="text-xs font-normal text-slate-400">ms/tok</span>
            </div>
            <div className="mt-1 font-mono text-xs text-slate-500">
              KV-cached per-step dot product
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      {chartMode === 'interactive' ? (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Memory SVG */}
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-900">
                State Tensor Bytes vs. KV Cache Bytes
              </h3>
              <span className="font-mono text-xs text-slate-500">KiB (fp32)</span>
            </div>
            <svg viewBox="0 0 540 240" className="w-full" role="img" aria-label="Memory bytes vs sequence length">
              {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
                const y = 198 - frac * 165;
                const val = (frac * maxKib).toFixed(0);
                return (
                  <g key={frac}>
                    <line x1={62} y1={y} x2={515} y2={y} stroke="#E2E8F0" strokeWidth={1} />
                    <text x={54} y={y + 4} textAnchor="end" className="fill-slate-400 font-mono text-[10px]">
                      {val}
                    </text>
                  </g>
                );
              })}

              {(() => {
                const xs = [125, 300, 475];
                const mPts = lengths.map((l, i) => ({
                  x: xs[i],
                  y: 198 - (mambaBytes[i] / 1024 / maxKib) * 165,
                  kib: mambaBytes[i] / 1024,
                  l,
                }));
                const tPts = lengths.map((l, i) => ({
                  x: xs[i],
                  y: 198 - (tfBytes[i] / 1024 / maxKib) * 165,
                  kib: tfBytes[i] / 1024,
                  l,
                }));
                const path = (pts: { x: number; y: number }[]) =>
                  pts.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

                return (
                  <g>
                    <path d={path(mPts)} fill="none" stroke="#2563EB" strokeWidth={2.5} />
                    <path d={path(tPts)} fill="none" stroke="#D97706" strokeWidth={2.5} />
                    {mPts.map((p) => (
                      <g key={`mb-${p.l}`}>
                        <circle cx={p.x} cy={p.y} r={4.5} fill="#2563EB" stroke="#FFF" strokeWidth={1.5} />
                        <text x={p.x} y={p.y - 9} textAnchor="middle" className="fill-blue-700 font-mono text-[10px] font-semibold">
                          {p.kib.toFixed(1)} KiB
                        </text>
                      </g>
                    ))}
                    {tPts.map((p) => (
                      <g key={`tb-${p.l}`}>
                        <rect x={p.x - 4} y={p.y - 4} width={8} height={8} fill="#D97706" stroke="#FFF" strokeWidth={1.5} />
                        <text x={p.x - 6} y={Math.max(16, p.y - 10)} textAnchor="middle" className="fill-amber-700 font-mono text-[10px] font-semibold">
                          {p.kib.toFixed(0)} KiB
                        </text>
                      </g>
                    ))}
                    {xs.map((x, idx) => (
                      <text key={lengths[idx]} x={x} y={222} textAnchor="middle" className="fill-slate-600 font-mono text-xs">
                        L={lengths[idx]}
                      </text>
                    ))}
                  </g>
                );
              })()}
            </svg>
          </div>

          {/* Latency SVG */}
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-900">
                Per-Token Autoregressive Stepping Latency
              </h3>
              <span className="font-mono text-xs text-slate-500">ms / token</span>
            </div>
            <svg viewBox="0 0 540 240" className="w-full" role="img" aria-label="Per-token latency vs sequence length">
              {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
                const y = 198 - frac * 165;
                const val = (frac * maxMs).toFixed(2);
                return (
                  <g key={frac}>
                    <line x1={62} y1={y} x2={515} y2={y} stroke="#E2E8F0" strokeWidth={1} />
                    <text x={54} y={y + 4} textAnchor="end" className="fill-slate-400 font-mono text-[10px]">
                      {val}
                    </text>
                  </g>
                );
              })}

              {(() => {
                const xs = [125, 300, 475];
                const mPts = lengths.map((l, i) => ({
                  x: xs[i],
                  y: 198 - (mambaMs[i] / maxMs) * 165,
                  ms: mambaMs[i],
                  l,
                }));
                const tPts = lengths.map((l, i) => ({
                  x: xs[i],
                  y: 198 - (tfMs[i] / maxMs) * 165,
                  ms: tfMs[i],
                  l,
                }));
                const path = (pts: { x: number; y: number }[]) =>
                  pts.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

                return (
                  <g>
                    <path d={path(mPts)} fill="none" stroke="#2563EB" strokeWidth={2.5} />
                    <path d={path(tPts)} fill="none" stroke="#D97706" strokeWidth={2.5} />
                    {mPts.map((p) => (
                      <g key={`ml-${p.l}`}>
                        <circle cx={p.x} cy={p.y} r={4.5} fill="#2563EB" stroke="#FFF" strokeWidth={1.5} />
                        <text x={p.x} y={Math.max(16, p.y - 9)} textAnchor="middle" className="fill-blue-700 font-mono text-[10px] font-semibold">
                          {p.ms.toFixed(3)} ms
                        </text>
                      </g>
                    ))}
                    {tPts.map((p) => (
                      <g key={`tl-${p.l}`}>
                        <rect x={p.x - 4} y={p.y - 4} width={8} height={8} fill="#D97706" stroke="#FFF" strokeWidth={1.5} />
                        <text x={p.x} y={Math.min(212, p.y + 16)} textAnchor="middle" className="fill-amber-700 font-mono text-[10px] font-semibold">
                          {p.ms.toFixed(3)} ms
                        </text>
                      </g>
                    ))}
                    {xs.map((x, idx) => (
                      <text key={lengths[idx]} x={x} y={222} textAnchor="middle" className="fill-slate-600 font-mono text-xs">
                        L={lengths[idx]}
                      </text>
                    ))}
                  </g>
                );
              })()}
            </svg>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-2 text-xs font-mono text-slate-500">reports/inference_memory.png</div>
            <img
              src="/reports/inference_memory.png"
              alt="Matplotlib inference memory plot comparing Mamba state vs Transformer KV cache"
              referrerPolicy="no-referrer"
              className="w-full rounded border border-slate-100"
            />
          </div>
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-2 text-xs font-mono text-slate-500">reports/inference_latency.png</div>
            <img
              src="/reports/inference_latency.png"
              alt="Matplotlib per-token latency plot comparing Mamba vs Transformer"
              referrerPolicy="no-referrer"
              className="w-full rounded border border-slate-100"
            />
          </div>
        </div>
      )}

      {/* Interactive Token-by-Token Stepping State Inspector */}
      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Autoregressive Stepping Tensor Inspector
            </h3>
            <p className="text-xs text-slate-500">
              Simulate token-by-token generation at batch=1, fp32. Mamba stores h (1, 128, 16) + conv tail (1, 128, 3); Transformer appends K, V (1, 4, t, 16) per layer.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setStepCount((s) => Math.min(8192, s + 1))}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
            >
              <StepForward className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Step +1</span>
            </button>
            <button
              type="button"
              onClick={() => setStepCount((s) => Math.min(8192, s + 256))}
              className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-700 whitespace-nowrap shrink-0"
            >
              <Play className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Step +256</span>
            </button>
            <button
              type="button"
              onClick={() => setStepCount(1)}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Reset</span>
            </button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="rounded border border-slate-200 bg-slate-50 p-3.5">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Tokens Generated
            </div>
            <div className="mt-1 font-mono text-xl font-bold text-slate-900 tabular-nums">
              {stepCount.toLocaleString()} tokens
            </div>
            <div className="mt-1 font-mono text-xs text-slate-500">
              Conv tail holds last 3 sequence vectors
            </div>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3.5">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-blue-700">
              Mamba Recurrent State
            </div>
            <div className="mt-1 font-mono text-xl font-bold text-slate-900 tabular-nums">
              {mambaStepBytes.toLocaleString()} B
            </div>
            <div className="mt-1 font-mono text-xs text-emerald-700">
              ● Constant O(1) · {(mambaStepBytes / 1024).toFixed(1)} KiB
            </div>
          </div>

          <div className="rounded border border-slate-200 bg-slate-50 p-3.5">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-amber-700">
              Transformer KV Cache
            </div>
            <div className="mt-1 font-mono text-xl font-bold text-slate-900 tabular-nums">
              {tfStepBytes.toLocaleString()} B
            </div>
            <div className="mt-1 font-mono text-xs text-amber-700">
              ▲ Linear O(L) · {(tfStepBytes / mambaStepBytes).toFixed(1)}x Mamba state
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
