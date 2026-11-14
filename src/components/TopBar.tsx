import React from 'react';
import { Download } from 'lucide-react';

interface TopBarProps {
  onExportReport: () => void;
  hasReport: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({ onExportReport, hasReport }) => {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-8 px-6 py-3.5">
        <a
          href="#scan-lab"
          className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
        >
          Mamba SSM Workbench
        </a>

        <nav aria-label="Primary sections" className="hidden md:flex items-center gap-7 text-sm font-medium text-slate-600">
          <a
            href="#scan-lab"
            className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
          >
            Scan Lab
          </a>
          <a
            href="#tasks-eval"
            className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
          >
            Tasks &amp; Eval
          </a>
          <a
            href="#memory-bench"
            className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
          >
            Memory Bench
          </a>
          <a
            href="#code-tests"
            className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
          >
            Code &amp; Tests
          </a>
        </nav>

        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={onExportReport}
            disabled={!hasReport}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Export Report</span>
          </button>
        </div>
      </div>
    </header>
  );
};
