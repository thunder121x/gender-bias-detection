import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';

const normalizeRange = (range) => {
  if (!range) return null;
  const [start, end] = range;
  return [Math.min(start, end), Math.max(start, end)];
};

const TokenAnnotator = ({
  tokens,
  rationales,
  currentRationale,
  mode,
  onAddSpan,
  onModeChange
}) => {
  const [selection, setSelection] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const previewRange = normalizeRange(selection);

  const { rationaleSet, triggerSet } = useMemo(() => {
    const rat = new Set();
    const trg = new Set();

    const markRange = (set, [start, end]) => {
      if (start === undefined || end === undefined) return;
      for (let i = start; i <= end; i += 1) set.add(i);
    };

    (rationales || []).forEach((r) => {
      (r.spans || []).forEach((range) => markRange(rat, range));
      (r.triggers || []).forEach((t) => markRange(trg, t.span || []));
    });
    (currentRationale.spans || []).forEach((range) => markRange(rat, range));
    (currentRationale.triggers || []).forEach((t) => markRange(trg, t.span || []));

    return { rationaleSet: rat, triggerSet: trg };
  }, [rationales, currentRationale]);

  useEffect(() => {
    const handleMouseUp = () => {
      if (isDragging) {
        commitSelection();
      }
    };
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [isDragging, previewRange, onAddSpan]);

  const commitSelection = () => {
    if (previewRange) {
      onAddSpan(previewRange);
    }
    setIsDragging(false);
    setSelection(null);
  };

  const beginSelection = (idx) => {
    setIsDragging(true);
    setSelection([idx, idx]);
  };

  const extendSelection = (idx) => {
    if (!isDragging) return;
    setSelection(([start]) => [start, idx]);
  };

  const isInPreview = (idx) =>
    previewRange && idx >= previewRange[0] && idx <= previewRange[1];

  const getTokenStyle = (idx) => {
    if (isInPreview(idx)) {
      return mode === 'trigger' ? 'bg-trigger-light' : 'bg-rationale-light';
    }
    if (triggerSet.has(idx)) return 'bg-trigger text-white';
    if (rationaleSet.has(idx)) return 'bg-rationale text-white';
    return 'bg-slate-100';
  };

  return (
    <div className="scroll-card p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm text-slate-600">
          Drag to select tokens. Current mode:{' '}
          <span className="font-semibold text-rose-600 uppercase">{mode}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <button
            type="button"
            onClick={() => onModeChange?.('rationale')}
            className="px-2 py-1 rounded bg-rationale-light text-rose-700 border border-transparent hover:border-rose-200 transition"
          >
            Rationale
          </button>
          <button
            type="button"
            onClick={() => onModeChange?.('trigger')}
            className="px-2 py-1 rounded bg-trigger-light text-red-700 border border-transparent hover:border-red-200 transition"
          >
            Trigger
          </button>
        </div>
      </div>
      <div className="flex flex-wrap gap-2 p-3 bg-slate-50 rounded-lg border border-slate-200">
        {tokens.map((token, idx) => (
          <span
            key={`${token}-${idx}`}
            className={clsx(
              'token-span px-2 py-1 rounded cursor-pointer select-none',
              getTokenStyle(idx)
            )}
            onMouseDown={() => beginSelection(idx)}
            onMouseEnter={() => extendSelection(idx)}
            onMouseUp={commitSelection}
          >
            {token}
          </span>
        ))}
      </div>
    </div>
  );
};

export default TokenAnnotator;
