const ProgressBar = ({ current, total }) => {
  const percentage = total ? Math.round((current / total) * 100) : 0;
  return (
    <div className="w-full mb-4">
      <div className="flex items-center justify-between text-sm text-slate-600 mb-1">
        <span>
          Sentence {Math.min(current, total)} of {total || 0}
        </span>
        <span>{percentage}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-rose-400 to-red-500 transition-all"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export default ProgressBar;
