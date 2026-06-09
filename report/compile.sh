#!/bin/bash
# LaTeX to PDF compilation script for main_full.tex

TEXFILE="${1:-main_full.tex}"
TMPDIR=$(mktemp -d)

echo "📄 Compiling: $TEXFILE"
xelatex -interaction=nonstopmode -output-directory="$TMPDIR" "$TEXFILE" > /dev/null 2>&1 && \
xelatex -interaction=nonstopmode -output-directory="$TMPDIR" "$TEXFILE" > /dev/null 2>&1 && \
cp "$TMPDIR/${TEXFILE%.tex}.pdf" . && \
rm -rf "$TMPDIR" && \
echo "✅ Success: ${TEXFILE%.tex}.pdf created" || \
echo "❌ Compilation failed"
