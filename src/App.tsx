import { useEffect, useState } from 'react';
import { TopBar } from './components/TopBar';
import { ScanLabSection } from './components/ScanLabSection';
import { TasksEvalSection } from './components/TasksEvalSection';
import { MemoryBenchSection } from './components/MemoryBenchSection';
import { CodeVerificationSection } from './components/CodeVerificationSection';
import {
  EmpiricalReportsBundle,
  InferenceReport,
  ScanEquivalenceReport,
  TrainEvalReport,
} from './types/reports';

export default function App() {
  const [scanEquivalence, setScanEquivalence] = useState<ScanEquivalenceReport | null>(null);
  const [trainEval, setTrainEval] = useState<TrainEvalReport | null>(null);
  const [inference, setInference] = useState<InferenceReport | null>(null);
  const [sourceFiles, setSourceFiles] = useState<Record<string, string>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchJsonSafe<T>(url: string): Promise<T | null> {
      try {
        const res = await fetch(url);
        const contentType = res.headers.get('content-type') || '';
        if (!res.ok || !contentType.includes('application/json')) {
          return null;
        }
        return (await res.json()) as T;
      } catch {
        return null;
      }
    }

    async function loadReports() {
      try {
        const [scanData, evalData, inferData, srcData] = await Promise.all([
          fetchJsonSafe<ScanEquivalenceReport>('/reports/scan_equivalence.json'),
          fetchJsonSafe<TrainEvalReport>('/reports/train_eval.json'),
          fetchJsonSafe<InferenceReport>('/reports/inference.json'),
          fetchJsonSafe<Record<string, string>>('/reports/sources.json'),
        ]);

        if (!cancelled) {
          setScanEquivalence(scanData);
          setTrainEval(evalData);
          setInference(inferData);
          setSourceFiles(srcData ?? {});
          if (!scanData && !inferData) {
            setLoadError('Run bench_infer.py and unit tests to populate report artifacts.');
          }
        }
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : 'Unable to load reports.');
        }
      }
    }

    loadReports();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleExportReport = () => {
    if (!scanEquivalence || !inference) return;
    const bundle: Partial<EmpiricalReportsBundle> = {
      scanEquivalence,
      ...(trainEval ? { trainEval } : {}),
      inference,
      sourceFiles,
    };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mamba_ssm_empirical_report.json';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <TopBar
        onExportReport={handleExportReport}
        hasReport={Boolean(scanEquivalence && inference)}
      />

      <main className="mx-auto max-w-[1400px] space-y-12 px-6 py-8">
        {loadError && (
          <div
            role="alert"
            className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-xs font-medium text-amber-900"
          >
            Empirical report bundle notice: {loadError}
          </div>
        )}

        <ScanLabSection />
        <TasksEvalSection trainEval={trainEval} onUploadTrainEval={setTrainEval} />
        <MemoryBenchSection inference={inference} />
        <CodeVerificationSection scanEquivalence={scanEquivalence} sourceFiles={sourceFiles} />
      </main>

      <footer className="mt-12 border-t border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-4 px-6 py-5 text-xs text-slate-500">
          <span>Mamba Selective SSM Workbench · Pure PyTorch ZOH &amp; Parallel Associative Scan</span>
          <span className="font-mono">d_model=64 · n_layers=2 · d_state=16 · expand=2 · conv_kernel=4</span>
        </div>
      </footer>
    </div>
  );
}
