import { useEffect, useState } from 'react';

const STORAGE_KEY = 'annotator-state-v1';

const emptyRationale = () => ({
  id: '',
  spans: [],
  triggers: []
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
      setSentences(parsed.sentences || []);
      setCurrentIndex(parsed.currentIndex || 0);
      setView(parsed.view || 'upload');
      setCurrentRationale(parsed.currentRationale || emptyRationale());
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
        rationales: row.rationales ?? []
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
