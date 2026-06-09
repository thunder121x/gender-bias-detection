# synthesizer_v2

Generates synthetic Thai social-media text for gender-bias detection research.
Label taxonomy follows `annotation-guideline.md` (Binary + 3 Subtypes).

## Install

```bash
# from project root
uv pip install -e "services/synthesizer_v2"
```

## Quick start

The tool auto-detects your API key and endpoint from `.env` (project root or
`services/synthesizer_v2/.env`).

```bash
# .env (project root)
GEMINI_API_KEY=AIza...          # preferred — routes to Google AI Studio
OPENROUTER_API_KEY=sk-or-...    # fallback — routes to OpenRouter
```

Then just run:

```bash
synth-v2 --mode gb-attack --count 100 --output output/gb_attack.json
```

No `--model` or `--base-url` needed when `GEMINI_API_KEY` is set — defaults are
already configured for Google AI Studio.

## Generate 4 000 samples per class

```bash
#!/usr/bin/env bash
# Run from anywhere — .env is loaded from cwd automatically
set -euo pipefail

COUNT=4000

synth-v2 --mode gb-attack      --count $COUNT --output output/gb_attack.json
synth-v2 --mode gb-normative   --count $COUNT --output output/gb_normative.json
synth-v2 --mode gb-sex         --count $COUNT --output output/gb_sex.json
synth-v2 --mode non-gb-neutral --count $COUNT --output output/non_gb_neutral.json
synth-v2 --mode non-gb-meta    --count $COUNT --output output/non_gb_meta.json
synth-v2 --mode non-gb-insult  --count $COUNT --output output/non_gb_insult.json
```

Output files are written to `services/synthesizer_v2/output/` by default.
Each run **resumes automatically** from a checkpoint (`.ckpt.json`) if interrupted.

## CLI reference

```
synth-v2 --mode MODE [options]
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `gb-attack` · `gb-normative` · `gb-sex` · `non-gb-neutral` · `non-gb-meta` · `non-gb-insult` |
| `--count` | `100` | Total items to generate |
| `--output` | `output/<mode>.json` | Output JSON file path |
| `--model` | `gemini-2.5-flash-lite-preview-06-17` | Model ID |
| `--base-url` | auto | API base URL (auto-set from key type) |
| `--api-key` | *(from .env)* | Override API key |
| `--temperature` | `0.95` | Sampling temperature |
| `--max-tokens` | `8192` | Max output tokens per call |
| `--seed` | *(none)* | Random seed for reproducibility |
| `--dry-run` | `false` | Print prompts only, no API call |

## Label taxonomy

| Mode | `label` | `subtype` |
|---|---|---|
| `gb-attack` | `GB` | `GB-ATTACK` |
| `gb-normative` | `GB` | `GB-NORMATIVE` |
| `gb-sex` | `GB` | `GB-SEX` |
| `non-gb-neutral` | `NON-GB` | `neutral` |
| `non-gb-meta` | `NON-GB` | `meta_counter` |
| `non-gb-insult` | `NON-GB` | `gendered_insult` |

## Output format

```json
[
  {
    "text": "ข้อความโซเชียลมีเดียภาษาไทย",
    "label": "GB",
    "subtype": "GB-ATTACK",
    "bias_target": "ผู้หญิง"
  }
]
```

`NON-GB` items do not have a `bias_target` field.

## Using a specific model / endpoint

```bash
# Google AI Studio (default when GEMINI_API_KEY is set)
synth-v2 --mode gb-attack --count 500 \
  --model gemini-2.5-flash-lite-preview-06-17 \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai \
  --output output/gb_attack.json

# OpenRouter
synth-v2 --mode gb-attack --count 500 \
  --model google/gemini-2.5-flash-lite \
  --base-url https://openrouter.ai/api/v1 \
  --output output/gb_attack.json
```
