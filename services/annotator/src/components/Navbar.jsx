const Navbar = () => (
  <header className="bg-white shadow-sm">
    <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-xl font-semibold text-slate-800">Annotator Tool</span>
      </div>
      <div className="text-sm text-slate-500">Browser-only · CSV in / CSV out</div>
    </div>
  </header>
);

export default Navbar;
