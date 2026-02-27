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
  disableExistingLabelPrompt = false,
  onReviseExistingLabel,
  onNextSentence,
  onPrevSentence
}) => {
  const [message, setMessage] = useState('');
  const [showExistingLabelPrompt, setShowExistingLabelPrompt] = useState(false);

  const LABEL_CYCLE = ['GB-NORMATIVE', 'GB-ATTACK', 'GB-SEX', 'NON-GB'];

  const cycleLabelType = () => {
    setCurrentRationale((prev) => {
      const currentIndex = LABEL_CYCLE.indexOf(prev.label_type);
      const nextIndex =
        currentIndex === -1
          ? 0
          : (currentIndex + 1) % LABEL_CYCLE.length;
      return { ...prev, label_type: LABEL_CYCLE[nextIndex] };
    });
  };

  const toggleDecisionRule = (rule) => {
    setCurrentRationale((prev) => {
      const exists = prev.decision_rule?.includes(rule);
      return {
        ...prev,
        decision_rule: exists
          ? prev.decision_rule.filter((x) => x !== rule)
          : [...(prev.decision_rule || []), rule]
      };
    });
  };

  const rangeToIndexList = (range) => {
    const [start, end] = range;
    const list = [];
    for (let i = Math.min(start, end); i <= Math.max(start, end); i += 1) list.push(i);
    return list;
  };

  const dedupeList = (list) =>
    Array.from(new Set((list || []).map((n) => Number(n)))).sort((a, b) => a - b);

  const handleAddSpan = (range) => {
    setMessage('');
    const asList = rangeToIndexList(range);
    if (mode === 'trigger') {
      if (!currentRationale.id) {
        setMessage('Set a rationale ID before adding triggers.');
        return;
      }
      setCurrentRationale((prev) => ({
        ...prev,
        triggers: dedupeList([...(prev.triggers || []), ...asList])
      }));
    } else {
      setCurrentRationale((prev) => ({
        ...prev,
        id: prev.id || nextRationaleId, // auto-assign if empty
        spans: dedupeList([...(prev.spans || []), ...asList])
      }));
    }
  };
  const nextRationaleId = useMemo(() => {
    const count = sentence.rationales?.length || 0;
    return `R${count}`;
  }, [sentence.rationales]);

  const handleFinishRationale = () => {
    if (!currentRationale.id) {
      setMessage('Rationale ID is required.');
      return;
    }
    if (!currentRationale.label_type) {
      setMessage('Select a bias type.');
      return;
    }
    if (!currentRationale.spans?.length) {
      setMessage('Add at least one rationale span.');
      return;
    }
    if (!currentRationale.triggers?.length) {
      setMessage('Add at least one trigger span.');
      return;
    }
    if (
      currentRationale.label_type === "NON-GB" &&
      (currentRationale.decision_rule?.includes("A") ||
        currentRationale.decision_rule?.includes("B"))
    ) {
      setMessage(`${currentRationale.label_type} cannot be A or B`);
      return;
    }

    if (
      currentRationale.label_type !== "NON-GB" &&
      (currentRationale.decision_rule?.includes("C") ||
        currentRationale.decision_rule?.includes("D"))
    ) {
      setMessage(`${currentRationale.label_type} cannot be C or D`);
      return;
    }
    onUpdateRationales((prev) => [
      ...prev,
      {
        ...currentRationale,
        spans: [currentRationale.spans || []],
        triggers: [currentRationale.triggers || []]
      }
    ]);
    setCurrentRationale({
      id: '',
      label_type: null,
      spans: [],
      triggers: [],
      decision_rule: []
    });
    setMode('rationale');
    setMessage('Rationale saved.');
  };

  const handleUndo = () => {
    onUpdateRationales((prev) => prev.slice(0, -1));
  };

  const handleClearCurrent = () => {
    setCurrentRationale({
      id: '',
      label_type: null,
      spans: [],
      triggers: [],
      decision_rule: []
    });
    setMessage('');
    setMode('rationale');
  };

  useEffect(() => {
    if (disableExistingLabelPrompt) {
      setShowExistingLabelPrompt(false);
      return;
    }
    setShowExistingLabelPrompt((sentence.rationales || []).length > 0);
  }, [sentence.id, disableExistingLabelPrompt]);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      const key = e.key.toLowerCase();
      // if (e.ctrlKey) return;
      if (key === "r" && !e.ctrlKey) setMode("rationale");
      if (key === "t" && !e.ctrlKey) setMode("trigger");
      if (key === "n" && !e.ctrlKey) onNextSentence();
      if (key === "p" && !e.ctrlKey) onPrevSentence();
      if (key === "l" && !e.ctrlKey) cycleLabelType();
      if (key === "a" && !e.ctrlKey) toggleDecisionRule("A");
      if (key === "b" && !e.ctrlKey) toggleDecisionRule("B");
      if (key === 'c' && e.shiftKey) {
        handleClearCurrent();
        return;
      }
      if (key === "c" && !e.ctrlKey) toggleDecisionRule("C");
      if (key === "d" && !e.ctrlKey) toggleDecisionRule("D");
      if (key === "f" && !e.ctrlKey) handleFinishRationale();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [
    setMode,
    onNextSentence,
    onPrevSentence,
    cycleLabelType,
    toggleDecisionRule,
    handleClearCurrent,
    handleFinishRationale
  ]);

  const savedPreview = useMemo(
    () => ({
      spans: [...(sentence.rationales || []).flatMap((r) => r.spans || [])],
      triggers: [...(sentence.rationales || []).flatMap((r) => r.triggers || [])],
      label_type: [...(sentence.rationales || []).map((r) => r.label_type).filter(Boolean)],
      decision_rule: [
        ...(sentence.rationales || [])
          .map((r) => r.decision_rule || [])
          .filter((rules) => (rules || []).length > 0)
      ]
    }),
    [sentence]
  );

  const flattenLists = (lists) =>
    (lists || [])
      .flatMap((entry) => (Array.isArray(entry) ? entry : [entry]))
      .map((n) => Number(n));

  const currentPreview = useMemo(
    () => ({
      spans: flattenLists(currentRationale.spans),
      triggers: flattenLists(currentRationale.triggers),
      label_type: currentRationale.label_type || null,
      decision_rule: currentRationale.decision_rule || []
    }),
    [currentRationale]
  );

  const biasColorClasses = (bias) => {
    if (bias === 'stereotype') return 'bg-blue-50 border-blue-200';
    if (bias === 'sexism') return 'bg-rose-50 border-rose-200';
    return 'bg-slate-50 border-slate-200';
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="scroll-card p-4">
          <div className="flex items-start justify-between">
            {/* Left Section */}
            <div>
              <div className="text-sm text-slate-500">
                Sentence {currentIndex + 1}
              </div>
              <h2 className="text-2xl font-bold text-slate-800 leading-tight">
                Annotate Tokens
              </h2>
            </div>

            {/* Shortcut Section */}
            <div className="text-xs text-slate-600 space-y-1 text-right max-w-xs">

              <div className="font-semibold text-slate-700">Shortcuts</div>

              <div className="flex flex-wrap gap-1 justify-end">
                <span className="px-2 py-0.5 bg-slate-100 rounded text-slate-700">R = rationale</span>
                <span className="px-2 py-0.5 bg-slate-100 rounded text-slate-700">T = trigger</span>
                <span className="px-2 py-0.5 bg-slate-100 rounded text-slate-700">L = cycle label</span>
              </div>

              <div className="flex flex-wrap gap-1 justify-end">
                <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded">A/B/C/D = decision</span>
                <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded">F = finish</span>
              </div>

              <div className="flex flex-wrap gap-1 justify-end">
                <span className="px-2 py-0.5 bg-rose-100 text-rose-700 rounded">Shift+C = clear</span>
                <span className="px-2 py-0.5 bg-slate-200 text-slate-700 rounded">N = next</span>
                <span className="px-2 py-0.5 bg-slate-200 text-slate-700 rounded">P = prev</span>
              </div>

            </div>
          </div>

          <p className="mt-3 text-slate-700 text-lg leading-relaxed">
            {sentence.text}
          </p>
        </div>

        <TokenAnnotator
          tokens={sentence.tokens}
          rationales={sentence.rationales}
          currentRationale={currentRationale}
          mode={mode}
          onModeChange={setMode}
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
        <div
          className={`scroll-card p-4 space-y-3 border ${biasColorClasses(
            currentRationale.label_type
          )}`}
        >
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-800">Current Rationale</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setMode("rationale")}
                className={`px-3 py-1 rounded-lg text-sm ${
                  mode === "rationale"
                    ? "bg-rationale text-white"
                    : "bg-slate-100 text-slate-700"
                }`}
              >
                Rationale Mode (R)
              </button>
              <button
                onClick={() => setMode("trigger")}
                className={`px-3 py-1 rounded-lg text-sm ${
                  mode === "trigger"
                    ? "bg-trigger text-white"
                    : "bg-slate-100 text-slate-700"
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

          <div>
            <div className="text-sm font-medium text-slate-700 mb-2">
              Label type
            </div>
            <div className="flex gap-2">
              {/* GB-NORMATIVE */}
              <button
                onClick={() =>
                  setCurrentRationale((prev) => ({
                    ...prev,
                    label_type: "GB-NORMATIVE",
                  }))
                }
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.label_type === "GB-NORMATIVE"
                    ? "bg-blue-100 border-blue-300 text-blue-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-blue-50"
                }`}
              >
                GB-NORMATIVE
              </button>

              {/* GB-ATTACK */}
              <button
                onClick={() =>
                  setCurrentRationale((prev) => ({
                    ...prev,
                    label_type: "GB-ATTACK",
                  }))
                }
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.label_type === "GB-ATTACK"
                    ? "bg-rose-100 border-rose-300 text-rose-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-rose-50"
                }`}
              >
                GB-ATTACK
              </button>

              {/* GB-SEX */}
              <button
                onClick={() =>
                  setCurrentRationale((prev) => ({
                    ...prev,
                    label_type: "GB-SEX",
                  }))
                }
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.label_type === "GB-SEX"
                    ? "bg-purple-100 border-purple-300 text-purple-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-purple-50"
                }`}
              >
                GB-SEX
              </button>

              {/* NON-GB */}
              <button
                onClick={() =>
                  setCurrentRationale((prev) => ({
                    ...prev,
                    label_type: "NON-GB",
                  }))
                }
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.label_type === "NON-GB"
                    ? "bg-purple-100 border-purple-300 text-purple-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-purple-50"
                }`}
              >
                NON-GB
              </button>
            </div>
          </div>

          <div>
            <div className="text-sm font-medium text-slate-700 mb-2">
              Decision Framework (multiple allowed)
            </div>

            <div className="grid grid-cols-2 gap-2">
              {/* A — Gendered Target */}
              <button
                onClick={() => {
                  setCurrentRationale((prev) => {
                    const exists = prev.decision_rule?.includes("A");
                    return {
                      ...prev,
                      decision_rule: exists
                        ? prev.decision_rule.filter((x) => x !== "A")
                        : [...(prev.decision_rule || []), "A"],
                    };
                  });
                }}
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.decision_rule?.includes("A")
                    ? "bg-blue-100 border-blue-300 text-blue-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-blue-50"
                }`}
              >
                A — Gendered Target
                <span className="block text-xs text-slate-500">
                  อ้างอิงเพศ / เพศสภาพ / SOGI
                </span>
              </button>

              {/* B — Negative Evaluation */}
              <button
                onClick={() => {
                  setCurrentRationale((prev) => {
                    const exists = prev.decision_rule?.includes("B");
                    return {
                      ...prev,
                      decision_rule: exists
                        ? prev.decision_rule.filter((x) => x !== "B")
                        : [...(prev.decision_rule || []), "B"],
                    };
                  });
                }}
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.decision_rule?.includes("B")
                    ? "bg-rose-100 border-rose-300 text-rose-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-rose-50"
                }`}
              >
                B — Negative Evaluation
                <span className="block text-xs text-slate-500">
                  ลดคุณค่า / เหยียด / เหมารวมเพราะเพศ
                </span>
              </button>

              {/* C — Meta Commentary */}
              <button
                onClick={() => {
                  setCurrentRationale((prev) => {
                    const exists = prev.decision_rule?.includes("C");
                    return {
                      ...prev,
                      decision_rule: exists
                        ? prev.decision_rule.filter((x) => x !== "C")
                        : [...(prev.decision_rule || []), "C"],
                    };
                  });
                }}
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.decision_rule?.includes("C")
                    ? "bg-purple-100 border-purple-300 text-purple-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-purple-50"
                }`}
              >
                C — Meta Commentary
                <span className="block text-xs text-slate-500">
                  วิจารณ์อคติ ไม่ใช่ผู้พูดเหยียดเอง
                </span>
              </button>

              {/* D — Sexual Language Test */}
              <button
                onClick={() => {
                  setCurrentRationale((prev) => {
                    const exists = prev.decision_rule?.includes("D");
                    return {
                      ...prev,
                      decision_rule: exists
                        ? prev.decision_rule.filter((x) => x !== "D")
                        : [...(prev.decision_rule || []), "D"],
                    };
                  });
                }}
                className={`px-3 py-2 rounded-lg text-sm border transition ${
                  currentRationale.decision_rule?.includes("D")
                    ? "bg-yellow-100 border-yellow-300 text-yellow-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-yellow-50"
                }`}
              >
                D - Gender Insult
                <span className="block text-xs text-slate-500">
                  ด่าหรือวิพากษ์วิจารณ์โดยกล่าวถึงเพศ แต่ไม่เหยียดเพศ
                </span>
              </button>
            </div>
          </div>

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
                className={`border rounded-lg p-3 ${biasColorClasses(
                  rationale.label_type
                )}`}
              >
                <div className="flex items-center justify-between">
                  <div className="font-medium text-slate-800">
                    ID: {rationale.id} · Type:{" "}
                    <span
                      className={`inline-flex items-center px-2 py-1 rounded text-xs font-semibold ${
                        rationale.label_type === "stereotype"
                          ? "bg-blue-200 text-blue-900"
                          : rationale.label_type === "sexism"
                          ? "bg-rose-200 text-rose-900"
                          : "bg-slate-200 text-slate-800"
                      }`}
                    >
                      {rationale.label_type || "n/a"}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500">#{idx + 1}</div>
                </div>
                <div className="text-xs text-slate-600 mt-1">
                  Rationale spans: {rationale.spans?.length || 0} · Trigger
                  spans: {rationale.triggers?.length || 0}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Spans: {JSON.stringify(rationale.spans || [])}
                </div>
                <div className="text-xs text-slate-500">
                  Triggers: {JSON.stringify(rationale.triggers || [])}
                </div>
                <div className="text-xs text-slate-500">
                  Decision: {JSON.stringify(rationale.decision_rule || [])}
                </div>
                <button
                  type="button"
                  onClick={() => onReviseExistingLabel?.(idx)}
                  className="mt-2 text-xs text-rose-600 hover:text-rose-700"
                >
                  Revise this rationale
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="scroll-card p-4 text-sm text-slate-600">
          <div className="font-semibold text-slate-800 mb-1">
            Saved Sentence JSON
          </div>
          <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-auto text-xs">
            {JSON.stringify(
              {
                tokens: sentence.tokens,
                rationales: savedPreview.spans,
                triggers: savedPreview.triggers,
                label_type: savedPreview.label_type,
                decision_rule: savedPreview.decision_rule,
              },
              null,
              2
            )}
          </pre>
          <div className="font-semibold text-slate-800 mt-4 mb-1">
            Pending Rationale
          </div>
          <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-auto text-xs">
            {JSON.stringify(
              {
                rationales: currentPreview.spans,
                triggers: currentPreview.triggers,
                label_type: currentPreview.label_type,
                decision_rule: currentPreview.decision_rule,
              },
              null,
              2
            )}
          </pre>
        </div>
      </div>

      {showExistingLabelPrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/45 px-4">
          <div className="w-full max-w-md rounded-xl bg-white border border-slate-200 shadow-lg p-5 space-y-4">
            <div>
              <h3 className="text-lg font-semibold text-slate-800">
                Existing label found
              </h3>
              <p className="text-sm text-slate-600 mt-1">
                This sentence already has saved labels. Choose whether to add a
                new label or revise existing labels.
              </p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setShowExistingLabelPrompt(false)}
                className="px-3 py-2 rounded-lg bg-slate-900 text-white hover:bg-slate-800 transition"
              >
                Add New Label
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowExistingLabelPrompt(false);
                  onReviseExistingLabel?.(0);
                }}
                className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
              >
                Revise Label
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnnotatePage;
