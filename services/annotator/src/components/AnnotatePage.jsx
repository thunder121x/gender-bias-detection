import { useEffect, useMemo, useState } from 'react';
import TokenAnnotator from './TokenAnnotator';

const AnnotatePage = ({
  sentence,
  currentIndex,
  total,
  mode,
  setMode,
  currentRationale,
  setCurrentRationale,
  onUpdateRationales,
  onFinishSentence,
  onNextSentence,
  onPrevSentence
}) => {
  const [message, setMessage] = useState('');

  useEffect(() => {
    const handleKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key.toLowerCase() === 'r') setMode('rationale');
      if (e.key.toLowerCase() === 't') setMode('trigger');
      if (e.key.toLowerCase() === 'n') onNextSentence();
      if (e.key.toLowerCase() === 'p') onPrevSentence();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [setMode, onNextSentence, onPrevSentence]);

  const handleAddSpan = (range) => {
    setMessage('');
    if (mode === 'trigger') {
      if (!currentRationale.id) {
        setMessage('Set a rationale ID before adding triggers.');
        return;
      }
      setCurrentRationale((prev) => ({
        ...prev,
        triggers: [...(prev.triggers || []), { span: range }]
      }));
    } else {
      setCurrentRationale((prev) => ({
        ...prev,
        id: prev.id || nextRationaleId, // auto-assign if empty
        spans: [...(prev.spans || []), range],
      }));
    }
  };
  const nextRationaleId = useMemo(() => {
    const count = sentence.rationales?.length || 0;
    return `R${count}`;
  }, [sentence.rationales]);

  const handleFinishRationale = () => {
    // if (!currentRationale.id) {
    //   setMessage('Rationale ID is required.');
    //   return;
    // }
    // if (!currentRationale.spans?.length) {
    //   setMessage('Add at least one rationale span.');
    //   return;
    // }
    if (!currentRationale.triggers?.length) {
      setMessage('Add at least one trigger span.');
      return;
    }
    onUpdateRationales((prev) => [...prev, { ...currentRationale }]);
    setCurrentRationale({ id: '', spans: [], triggers: [] });
    const switchToRationale = () => {
      setMode("rationale");
      setCurrentRationale((prev) => ({
        ...prev,
        id: prev.id || nextRationaleId, // auto fill only if empty
      }));
    };
    setMessage('Rationale saved.');
  };

  const handleUndo = () => {
    onUpdateRationales((prev) => prev.slice(0, -1));
  };

  const handleClearCurrent = () => {
    setCurrentRationale({ id: '', spans: [], triggers: [] });
    setMessage('');
    setMode('rationale');
  };

  const flattened = useMemo(
    () => ({
      spans: [
        ...(sentence.rationales || []).flatMap((r) => r.spans || []),
        ...(currentRationale.spans || [])
      ],
      triggers: [
        ...(sentence.rationales || []).flatMap((r) => r.triggers || []),
        ...(currentRationale.triggers || [])
      ]
    }),
    [sentence, currentRationale]
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="scroll-card p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-slate-500">Sentence {currentIndex + 1}</div>
              <h2 className="text-xl font-semibold text-slate-800">Annotate Tokens</h2>
            </div>
            <div className="text-sm text-slate-500">
              Shortcuts: R (rationale) · T (trigger) · N (next) · P (prev)
            </div>
          </div>
          <p className="mt-2 text-slate-600">{sentence.text}</p>
        </div>

        <TokenAnnotator
          tokens={sentence.tokens}
          rationales={sentence.rationales}
          currentRationale={currentRationale}
          mode={mode}
          onAddSpan={handleAddSpan}
        />

        <div className="flex items-center gap-3">
          <button
            onClick={onFinishSentence}
            className="px-4 py-2 rounded-lg bg-slate-900 text-white hover:bg-slate-800 transition"
          >
            Finish Annotation for This Sentence
          </button>
          <button
            onClick={onNextSentence}
            className="px-3 py-2 rounded-lg bg-rose-500 text-white hover:bg-rose-600 transition"
          >
            Next (N)
          </button>
          <button
            onClick={onPrevSentence}
            className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
          >
            Previous (P)
          </button>
          <div className="text-sm text-slate-500 ml-auto">
            {currentIndex + 1} / {total}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="scroll-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-800">Current Rationale</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setMode('rationale')}
                className={`px-3 py-1 rounded-lg text-sm ${
                  mode === 'rationale'
                    ? 'bg-rationale text-white'
                    : 'bg-slate-100 text-slate-700'
                }`}
              >
                Rationale Mode (R)
              </button>
              <button
                onClick={() => setMode('trigger')}
                className={`px-3 py-1 rounded-lg text-sm ${
                  mode === 'trigger'
                    ? 'bg-trigger text-white'
                    : 'bg-slate-100 text-slate-700'
                }`}
              >
                Trigger Mode (T)
              </button>
            </div>
          </div>

          <label className="block text-sm font-medium text-slate-700">
            Rationale ID
            <input
              type="text"
              value={currentRationale.id}
              onChange={(e) =>
                setCurrentRationale((prev) => ({ ...prev, id: e.target.value }))
              }
              placeholder="Enter rationale id"
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-rose-300"
            />
          </label>

          <div className="text-sm text-slate-600">
            <div>Rationale spans: {currentRationale.spans?.length || 0}</div>
            <div>Trigger spans: {currentRationale.triggers?.length || 0}</div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleFinishRationale}
              className="px-3 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition"
            >
              Finish Rationale
            </button>
            <button
              onClick={handleClearCurrent}
              className="px-3 py-2 bg-white border border-slate-200 text-slate-700 rounded-lg hover:bg-slate-50 transition"
            >
              Clear Current
            </button>
          </div>

          {message && (
            <div className="text-sm text-rose-600 bg-rose-50 border border-rose-100 rounded-md p-2">
              {message}
            </div>
          )}
        </div>

        <div className="scroll-card p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-slate-800">Saved Rationales</h3>
            <button
              onClick={handleUndo}
              className="text-sm text-rose-600 hover:text-rose-700"
            >
              Undo last
            </button>
          </div>
          <div className="space-y-2">
            {(sentence.rationales || []).length === 0 && (
              <p className="text-sm text-slate-500">None yet.</p>
            )}
            {(sentence.rationales || []).map((rationale, idx) => (
              <div
                key={`${rationale.id}-${idx}`}
                className="border border-slate-200 rounded-lg p-3 bg-slate-50"
              >
                <div className="flex items-center justify-between">
                  <div className="font-medium text-slate-800">ID: {rationale.id}</div>
                  <div className="text-xs text-slate-500">#{idx + 1}</div>
                </div>
                <div className="text-xs text-slate-600 mt-1">
                  Rationale spans: {rationale.spans?.length || 0} · Trigger spans:{' '}
                  {rationale.triggers?.length || 0}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="scroll-card p-4 text-sm text-slate-600">
          <div className="font-semibold text-slate-800 mb-1">Current Sentence JSON</div>
          <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-auto text-xs">
            {JSON.stringify(
              {
                tokens: sentence.tokens,
                rationales: flattened.spans,
                triggers: flattened.triggers
              },
              null,
              2
            )}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default AnnotatePage;
