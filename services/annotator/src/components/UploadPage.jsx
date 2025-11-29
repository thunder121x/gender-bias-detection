import { useEffect } from 'react';
import { useCSVLoader } from '../hooks/useCSVLoader';

const UploadPage = ({ onLoaded }) => {
  const { rows, error, loading, loadFile, reset } = useCSVLoader();

  useEffect(() => {
    if (rows.length) {
      onLoaded(rows);
    }
  }, [rows, onLoaded]);

  return (
    <div className="scroll-card p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Upload CSV</h1>
          <p className="text-slate-600 mt-1">
            Provide a CSV with <code className="font-mono text-sm">id</code> and{' '}
            <code className="font-mono text-sm">tokens</code> columns to begin.
          </p>
        </div>
        <button
          onClick={reset}
          className="text-sm text-slate-500 hover:text-slate-700 transition"
        >
          Reset
        </button>
      </div>

      <label className="block border-2 border-dashed border-rose-200 rounded-xl p-10 text-center cursor-pointer hover:border-rose-300 transition bg-rose-50/40">
        <input
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => loadFile(e.target.files?.[0])}
        />
        <div className="text-lg font-medium text-rose-600">
          Click to upload or drag CSV file here
        </div>
        <div className="text-slate-500 mt-2">All processing happens in your browser.</div>
      </label>

      {loading && <p className="text-sm text-slate-600 mt-4">Reading file…</p>}
      {error && (
        <p className="text-sm text-red-500 mt-4 bg-red-50 border border-red-100 rounded-md p-3">
          {error}
        </p>
      )}
    </div>
  );
};

export default UploadPage;
