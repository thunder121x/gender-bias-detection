# Annotator Tool (React + Vite + Tailwind)

Browser-only CSV annotation tool for rationale/trigger spans.

## Quick start

```bash
cd services/annotator
npm install
npm run dev
```

Open the printed localhost URL.

## Workflow

1. Upload CSV with `id` and `tokens` (or `text`) columns.
2. Annotate each sentence with rationale spans (pink) and trigger spans (red). Drag across tokens to select.
3. Choose a bias type for the rationale: **Gender Stereotypes** or **Sexism / Derogatory**.
4. Finish each rationale (requires id + bias type + ≥1 rationale span + ≥1 trigger span).
4. Validate sentence, then move to next. Navigation shortcuts: `R` (rationale), `T` (trigger), `N` (next), `P` (previous).
5. After all sentences, export CSV with `id,text,tokens,rationales,triggers,bias_type`.

State auto-saves to `localStorage`; use “Start Over” on the export page to reset. All operations run client-side only.
