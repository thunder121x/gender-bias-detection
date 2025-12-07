import { buildCSV, downloadCSV } from '../utils/csvExporter';

const ExportPage = ({ sentences, onReset }) => {
  const handleDownload = () => {
    const csv = buildCSV(sentences);
    downloadCSV('annotations.csv', csv);
  };

  return (
    <div className="scroll-card p-8 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-slate-500">All sentences processed</div>
          <h2 className="text-2xl font-semibold text-slate-800">Export CSV</h2>
        </div>
        <div className="text-sm text-slate-600">
          {sentences.length} sentence{sentences.length === 1 ? '' : 's'}
        </div>
      </div>

      <p className="text-slate-600">
        Download the final annotations as CSV. Each row contains{' '}
        <code className="font-mono text-sm">id</code>,{' '}
        <code className="font-mono text-sm">text</code>,{' '}
        <code className="font-mono text-sm">tokens</code>,{' '}
        <code className="font-mono text-sm">rationales</code>,{' '}
        <code className="font-mono text-sm">triggers</code>,{' '}
        <code className="font-mono text-sm">label_type</code>, and{' '}
        <code className="font-mono text-sm">decision_rule</code>.
      </p>

      <div className="flex gap-3">
        <button
          onClick={handleDownload}
          className="px-4 py-2 rounded-lg bg-rose-500 text-white hover:bg-rose-600 transition"
        >
          Download CSV
        </button>
        <button
          onClick={onReset}
          className="px-4 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
        >
          Start Over
        </button>
      </div>

      <div className="border border-slate-200 rounded-lg p-4 bg-slate-50 text-sm text-slate-700">
        <div className="font-semibold text-slate-800 mb-2">Preview</div>
        <pre className="overflow-auto text-xs">
          {buildCSV(sentences.slice(0, 3)).split('\n').slice(0, 6).join('\n')}
        </pre>
      </div>
    </div>
  );
};

export default ExportPage;
