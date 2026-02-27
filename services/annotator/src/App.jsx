import { useMemo, useState } from 'react';
import UploadPage from './components/UploadPage';
import AnnotatePage from './components/AnnotatePage';
import ValidatePage from './components/ValidatePage';
import ReviseLabelPage from './components/ReviseLabelPage';
import ExportPage from './components/ExportPage';
import Navbar from './components/Navbar';
import ProgressBar from './components/ProgressBar';
import { useAnnotationState } from './hooks/useAnnotationState';

const App = () => {
  const {
    view,
    sentences,
    currentIndex,
    currentRationale,
    setCurrentRationale,
    mode,
    setMode,
    loadSentences,
    updateRationalesForCurrent,
    goToNextSentence,
    goToPreviousSentence,
    setView,
    resetState
  } = useAnnotationState();
  const [reviseRationaleIndex, setReviseRationaleIndex] = useState(0);

  const currentSentence = useMemo(
    () => sentences[currentIndex] || null,
    [sentences, currentIndex]
  );

  return (
    <div className="min-h-screen">
      <Navbar />
      {view !== 'upload' && sentences.length > 0 && (
        <div className="max-w-6xl mx-auto px-4 mt-4">
          <ProgressBar current={currentIndex + 1} total={sentences.length} />
        </div>
      )}
      <main className="max-w-6xl mx-auto px-4 py-6">
        {view === 'upload' && <UploadPage onLoaded={loadSentences} />}

        {view === 'annotate' && currentSentence && (
          <AnnotatePage
            sentence={currentSentence}
            currentIndex={currentIndex}
            total={sentences.length}
            mode={mode}
            setMode={setMode}
            currentRationale={currentRationale}
            setCurrentRationale={setCurrentRationale}
            onUpdateRationales={updateRationalesForCurrent}
            onFinishSentence={() => setView('validate')}
            onReviseExistingLabel={(rationaleIndex = 0) => {
              setReviseRationaleIndex(rationaleIndex);
              setView('revise');
            }}
            onNextSentence={goToNextSentence}
            onPrevSentence={goToPreviousSentence}
          />
        )}

        {view === 'revise' && currentSentence && (
          <ReviseLabelPage
            sentence={currentSentence}
            initialRationaleIndex={reviseRationaleIndex}
            onUpdateRationales={updateRationalesForCurrent}
            onBack={() => setView('annotate')}
            onDone={() => setView('validate')}
          />
        )}

        {view === 'validate' && currentSentence && (
          <ValidatePage
            sentence={currentSentence}
            onBack={() => setView('annotate')}
            onConfirm={() => goToNextSentence()}
          />
        )}

        {view === 'export' && (
          <ExportPage sentences={sentences} onReset={() => resetState()} />
        )}
      </main>
    </div>
  );
};

export default App;
