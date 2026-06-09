# Senior Project Report

LaTeX source for the project report. Compiled output included as
[main_full.pdf](main_full.pdf).

## Files

- `main_full.tex` — full report source (single file, English with Thai abstract)
- `cpe-english-project.cls` — KMUTT CPE report document class
- `assets/images/` — figures (incl. `chapter4/` result figures)
- `assets/THSarabunNew/` — Thai fonts used via fontspec
- `puenc-greek.def` — hyperref helper
- `ref_appendix.md` — reference notes for the appendix
- `compile.sh` — build script

## Build

Requires **XeLaTeX** (TeX Live / MacTeX) and the Times New Roman system font.

```bash
cd report
./compile.sh          # runs xelatex twice, outputs main_full.pdf
# or manually:
xelatex main_full.tex && xelatex main_full.tex
```
