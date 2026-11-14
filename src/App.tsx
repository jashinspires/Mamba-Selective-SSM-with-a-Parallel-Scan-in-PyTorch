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

    async function loadReports() {
      try {
        const [scanRes, evalRes, inferRes, srcRes] = await Promise.all([
          fetch('/reports/scan_equivalence.json'),
          fetch('/reports/train_eval.json'),
          fetch('/reports/inference.json'),
          fetch('/reports/sources.json'),
        ]);

        if (!scanRes.ok || !evalRes.ok || !inferRes.ok || !srcRes.ok) {
          throw new Error('Failed to load empirical PyTorch report artifacts.');
        }

        const [scanData, evalData, inferData, srcData] = await Promise.all([
          scanRes.json() as Promise<ScanEquivalenceReport>,
          evalRes.json() as Promise<TrainEvalReport>,
          inferRes.json() as Promise<InferenceReport>,
          srcRes.json() as Promise<Record<string, string>>,
        ]);

        if (!cancelled) {
          setScanEquivalence(scanData);
          setTrainEval(evalData);
          setInference(inferData);
          setSourceFiles(srcData);
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
    if (!scanEquivalence || !trainEval || !inference) return;
    const bundle: EmpiricalReportsBundle = {
      scanEquivalence,
      trainEval,
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
        hasReport={Boolean(scanEquivalence && trainEval && inference)}
      />

      <main className="mx-auto max-w-[1400px] space-y-12 px-6 py-8">
        {loadError && (
          <div
            role="alert"
            className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-xs font-medium text-amber-900"
          >
            Empirical report bundle unavailable: {loadError}
          </div>
        )}

        <ScanLabSection />
        <TasksEvalSection trainEval={trainEval} />
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
