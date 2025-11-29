import { useMemo } from 'react';
import clsx from 'clsx';

const ValidatePage = ({ sentence, onBack, onConfirm }) => {
  const flattened = useMemo(() => {
    const rationaleSet = new Set();
    const triggerSet = new Set();
    const markRange = (set, [start, end]) => {
      if (start === undefined || end === undefined) return;
      for (let i = start; i <= end; i += 1) set.add(i);
    };
    (sentence.rationales || []).forEach((r) => {
      (r.spans || []).forEach((range) => markRange(rationaleSet, range));
      (r.triggers || []).forEach((t) => markRange(triggerSet, t.span || []));
    });
    return { rationaleSet, triggerSet };
  }, [sentence]);

  const exportPayload = useMemo(
    () => ({
      tokens: sentence.tokens,
      rationales: (sentence.rationales || []).flatMap((r) => r.spans || []),
      triggers: (sentence.rationales || []).flatMap((r) =>
        (r.triggers || []).map((t) => ({ span: t.span }))
      ),
      bias_type: (sentence.rationales || [])
        .map((r) => r.bias_type)
        .filter(Boolean)
    }),
    [sentence]
  );

  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${sentence.id}-annotation.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const getTokenClass = (idx) => {
    if (flattened.triggerSet.has(idx)) return 'bg-trigger text-white';
    if (flattened.rationaleSet.has(idx)) return 'bg-rationale text-white';
    return 'bg-slate-100';
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="scroll-card p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-slate-500">Validate Sentence</div>
              <h2 className="text-xl font-semibold text-slate-800">{sentence.id}</h2>
            </div>
            <div className="flex gap-2 text-xs text-slate-600">
              <span className="px-2 py-1 rounded bg-rationale-light text-rose-700">
                Rationale
              </span>
              <span className="px-2 py-1 rounded bg-trigger-light text-red-700">
                Trigger
              </span>
            </div>
          </div>
          <p className="mt-2 text-slate-600">{sentence.text}</p>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            {(exportPayload.bias_type || []).length === 0 ? (
              <span className="px-2 py-1 rounded bg-slate-100 text-slate-600">
                No bias type selected
              </span>
            ) : (
              exportPayload.bias_type.map((bias, idx) => (
                <span
                  key={`${bias}-${idx}`}
                  className={`px-2 py-1 rounded font-semibold ${
                    bias === 'stereotype'
                      ? 'bg-blue-200 text-blue-900'
                      : 'bg-rose-200 text-rose-900'
                  }`}
                >
                  {bias}
                </span>
              ))
            )}
          </div>
        </div>

        <div className="scroll-card p-4">
          <div className="flex flex-wrap gap-2 p-3 bg-slate-50 rounded-lg border border-slate-200">
            {sentence.tokens.map((token, idx) => (
              <span
                key={`${token}-${idx}`}
                className={clsx('px-2 py-1 rounded token-span', getTokenClass(idx))}
              >
                {token}
              </span>
            ))}
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onConfirm}
            className="px-4 py-2 rounded-lg bg-rose-500 text-white hover:bg-rose-600 transition"
          >
            Confirm & Next Sentence
          </button>
          <button
            onClick={onBack}
            className="px-4 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
          >
            Back to Annotate
          </button>
          <button
            onClick={downloadJSON}
            className="px-4 py-2 rounded-lg bg-slate-900 text-white hover:bg-slate-800 transition"
          >
            Download JSON
          </button>
        </div>
      </div>

      <div className="scroll-card p-4 text-sm text-slate-700">
        <div className="font-semibold text-slate-800 mb-2">Review JSON</div>
        <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-auto text-xs">
          {JSON.stringify(exportPayload, null, 2)}
        </pre>
      </div>
    </div>
  );
};

export default ValidatePage;
