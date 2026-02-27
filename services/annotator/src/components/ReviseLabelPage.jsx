import { useMemo, useState } from 'react';
import clsx from 'clsx';

const normalizeFlat = (lists) =>
  (lists || [])
    .flatMap((entry) => (Array.isArray(entry) ? entry : [entry]))
    .map((n) => Number(n))
    .filter((n) => Number.isFinite(n));

const ReviseLabelPage = ({
  sentence,
  initialRationaleIndex = 0,
  onUpdateRationales,
  onBack,
  onDone
}) => {
  const [selectedIndex, setSelectedIndex] = useState(initialRationaleIndex);
  const [mode, setMode] = useState('rationale');

  const rationales = sentence.rationales || [];
  const selected = rationales[selectedIndex] || null;

  const selectedRationaleSet = useMemo(
    () => new Set(normalizeFlat(selected?.spans)),
    [selected]
  );
  const selectedTriggerSet = useMemo(
    () => new Set(normalizeFlat(selected?.triggers)),
    [selected]
  );

  const handleTokenClick = (tokenIndex) => {
    if (!selected) return;
    onUpdateRationales((prev) =>
      prev.map((r, idx) => {
        if (idx !== selectedIndex) return r;
        if (mode === 'trigger') {
          const triggerList = new Set(normalizeFlat(r.triggers));
          if (triggerList.has(tokenIndex)) triggerList.delete(tokenIndex);
          else triggerList.add(tokenIndex);
          const next = Array.from(triggerList).sort((a, b) => a - b);
          return { ...r, triggers: [next] };
        }
        const spanList = new Set(normalizeFlat(r.spans));
        if (spanList.has(tokenIndex)) spanList.delete(tokenIndex);
        else spanList.add(tokenIndex);
        const next = Array.from(spanList).sort((a, b) => a - b);
        return { ...r, spans: [next] };
      })
    );
  };

  const handleRemoveRationale = () => {
    onUpdateRationales((prev) => prev.filter((_, idx) => idx !== selectedIndex));
    setSelectedIndex((idx) => Math.max(0, idx - 1));
  };

  const setSelectedLabelType = (labelType) => {
    if (!selected) return;
    onUpdateRationales((prev) =>
      prev.map((r, idx) =>
        idx === selectedIndex
          ? {
              ...r,
              label_type: labelType
            }
          : r
      )
    );
  };

  const toggleSelectedDecisionRule = (rule) => {
    if (!selected) return;
    onUpdateRationales((prev) =>
      prev.map((r, idx) => {
        if (idx !== selectedIndex) return r;
        const current = r.decision_rule || [];
        const exists = current.includes(rule);
        return {
          ...r,
          decision_rule: exists
            ? current.filter((x) => x !== rule)
            : [...current, rule]
        };
      })
    );
  };

  const getTokenClass = (idx) => {
    if (selectedTriggerSet.has(idx)) return 'bg-trigger text-white';
    if (selectedRationaleSet.has(idx)) return 'bg-rationale text-white';
    return 'bg-slate-100';
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="scroll-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-slate-500">Revise Existing Labels</div>
              <h2 className="text-xl font-semibold text-slate-800">{sentence.id}</h2>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setMode('rationale')}
                className={clsx(
                  'px-3 py-1 rounded-lg text-sm',
                  mode === 'rationale'
                    ? 'bg-rationale text-white'
                    : 'bg-slate-100 text-slate-700'
                )}
              >
                Rationale
              </button>
              <button
                onClick={() => setMode('trigger')}
                className={clsx(
                  'px-3 py-1 rounded-lg text-sm',
                  mode === 'trigger'
                    ? 'bg-trigger text-white'
                    : 'bg-slate-100 text-slate-700'
                )}
              >
                Trigger
              </button>
            </div>
          </div>
          <p className="text-slate-600">
            Click tokens to toggle them for the selected rationale.
          </p>
        </div>

        <div className="scroll-card p-4">
          <div className="flex flex-wrap gap-2 p-3 bg-slate-50 rounded-lg border border-slate-200">
            {(sentence.tokens || []).map((token, idx) => (
              <button
                key={`${token}-${idx}`}
                type="button"
                onClick={() => handleTokenClick(idx)}
                className={clsx('token-span px-2 py-1 rounded', getTokenClass(idx))}
              >
                {token}
              </button>
            ))}
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onDone}
            className="px-4 py-2 rounded-lg bg-rose-500 text-white hover:bg-rose-600 transition"
          >
            Done Revising
          </button>
          <button
            onClick={onBack}
            className="px-4 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
          >
            Back to Annotate
          </button>
        </div>
      </div>

      <div className="scroll-card p-4 space-y-2">
        <div className="font-semibold text-slate-800">Sentence Rationales</div>
        {rationales.length === 0 && (
          <div className="text-sm text-slate-500">No labels to revise.</div>
        )}
        {selected && (
          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 space-y-3">
            <div className="text-sm font-medium text-slate-800">
              Edit Selected Rationale
            </div>
            <div className="space-y-2">
              <div className="text-xs text-slate-600">Label Type</div>
              <div className="flex flex-wrap gap-2">
                {['GB-NORMATIVE', 'GB-ATTACK', 'GB-SEX', 'NON-GB'].map((label) => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => setSelectedLabelType(label)}
                    className={clsx(
                      'px-2 py-1 rounded text-xs border',
                      selected.label_type === label
                        ? 'bg-rose-100 border-rose-300 text-rose-800'
                        : 'bg-white border-slate-200 text-slate-700'
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-slate-600">Decision Rule</div>
              <div className="grid grid-cols-2 gap-2">
                {['A', 'B', 'C', 'D'].map((rule) => (
                  <button
                    key={rule}
                    type="button"
                    onClick={() => toggleSelectedDecisionRule(rule)}
                    className={clsx(
                      'px-2 py-1 rounded text-xs border',
                      (selected.decision_rule || []).includes(rule)
                        ? 'bg-blue-100 border-blue-300 text-blue-800'
                        : 'bg-white border-slate-200 text-slate-700'
                    )}
                  >
                    Rule {rule}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {rationales.map((rationale, idx) => (
          <div
            key={`${rationale.id || 'R'}-${idx}`}
            className={clsx(
              'border rounded-lg p-3 space-y-1',
              idx === selectedIndex
                ? 'border-rose-300 bg-rose-50'
                : 'border-slate-200 bg-white'
            )}
          >
            <button
              type="button"
              onClick={() => setSelectedIndex(idx)}
              className="w-full text-left"
            >
              <div className="font-medium text-slate-800">
                {(rationale.id || `R${idx}`)} · {rationale.label_type || 'n/a'}
              </div>
              <div className="text-xs text-slate-500">
                Decision: {JSON.stringify(rationale.decision_rule || [])}
              </div>
            </button>
            {idx === selectedIndex && (
              <button
                type="button"
                onClick={handleRemoveRationale}
                className="text-xs text-rose-600 hover:text-rose-700"
              >
                Remove this rationale
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReviseLabelPage;
