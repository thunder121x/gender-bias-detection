import { useEffect, useState } from 'react';

const STORAGE_KEY = 'annotator-state-v1';

const emptyRationale = () => ({
  id: '',
  bias_type: null,
  spans: [],
  triggers: []
});

const normalizeSpanList = (entry) => {
  if (Array.isArray(entry)) {
    if (entry.length === 2 && entry.every((n) => Number.isInteger(Number(n)))) {
      const start = Number(entry[0]);
      const end = Number(entry[1]);
      const list = [];
      for (let i = Math.min(start, end); i <= Math.max(start, end); i += 1) {
        list.push(i);
      }
      return list;
    }
    return entry.map((n) => Number(n));
  }
  if (entry && Array.isArray(entry.span)) {
    return normalizeSpanList(entry.span);
  }
  return [];
};

const hydrateRationale = (r) => ({
  bias_type: null,
  spans: [],
  triggers: [],
  ...r,
  spans: (r?.spans || []).map((s) => normalizeSpanList(s)),
  triggers: (r?.triggers || []).map((t) => normalizeSpanList(t))
});

export const useAnnotationState = () => {
  const [sentences, setSentences] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [view, setView] = useState('upload');
  const [currentRationale, setCurrentRationale] = useState(emptyRationale());
  const [mode, setMode] = useState('rationale');

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    try {
      const parsed = JSON.parse(saved);
      const hydrated = (parsed.sentences || []).map((s) => ({
        ...s,
        rationales: (s.rationales || []).map((r) => hydrateRationale(r))
      }));
      setSentences(hydrated);
      setCurrentIndex(parsed.currentIndex || 0);
      setView(parsed.view || 'upload');
      setCurrentRationale({ ...emptyRationale(), ...parsed.currentRationale });
      setMode(parsed.mode || 'rationale');
    } catch (err) {
      console.error('Failed to load saved state', err);
    }
  }, []);

  useEffect(() => {
    const payload = {
      sentences,
      currentIndex,
      view,
      currentRationale,
      mode
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }, [sentences, currentIndex, view, currentRationale, mode]);

  const loadSentences = (rows) => {
    setSentences(
      rows.map((row, idx) => ({
        id: row.id ?? `row-${idx + 1}`,
        tokens: row.tokens ?? [],
        text: row.text ?? (row.tokens ? row.tokens.join(' ') : ''),
        rationales: (row.rationales || []).map((r) => hydrateRationale(r))
      }))
    );
    setCurrentIndex(0);
    setView('annotate');
    setCurrentRationale(emptyRationale());
    setMode('rationale');
  };

  const updateRationalesForCurrent = (nextRationales) => {
    setSentences((prev) =>
      prev.map((sentence, idx) => {
        if (idx !== currentIndex) return sentence;
        const nextValue =
          typeof nextRationales === 'function'
            ? nextRationales(sentence.rationales || [])
            : nextRationales;
        return { ...sentence, rationales: nextValue };
      })
    );
  };

  const goToNextSentence = () => {
    if (!sentences.length) return;
    if (currentIndex < sentences.length - 1) {
      setCurrentIndex((idx) => idx + 1);
      setView('annotate');
    } else {
      setView('export');
    }
    setCurrentRationale(emptyRationale());
    setMode('rationale');
  };

  const goToPreviousSentence = () => {
    if (currentIndex === 0) return;
    setCurrentIndex((idx) => Math.max(0, idx - 1));
    setView('annotate');
    setCurrentRationale(emptyRationale());
    setMode('rationale');
  };

  const resetState = () => {
    setSentences([]);
    setCurrentIndex(0);
    setView('upload');
    setCurrentRationale(emptyRationale());
    setMode('rationale');
    localStorage.removeItem(STORAGE_KEY);
  };

  return {
    sentences,
    currentIndex,
    view,
    setView,
    mode,
    setMode,
    currentRationale,
    setCurrentRationale,
    loadSentences,
    updateRationalesForCurrent,
    goToNextSentence,
    goToPreviousSentence,
    resetState
  };
};
