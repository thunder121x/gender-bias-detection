# Models

Model weights are **not stored in git** (too large). Download them from the project
Google Drive and place them in this folder.

**Drive folder:** `<DRIVE_LINK>` <!-- paste the share link after uploading _to_drive/ -->

## Expected layout

```
models/
├── minilm_gender_bias_v2/          # fine-tuned MiniLM classifier (~450MB core files)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── crest_base_local/               # CREST base model (~1.1GB) — optional, only for retraining
└── gender_bias_qwen35_2b_unsloth/  # Qwen 3.5 2B LoRA adapter (~102MB)
```

## Used by

- `services/inference_app/app.py` — loads `models/minilm_gender_bias_v2` (override with `MODEL_PATH` env var)
- `services/finetuning/scripts/test_model.py` — same model, smoke test
- `services/lora_finetuning/inference_qwen_span.py` — Qwen span detector adapter

Only `minilm_gender_bias_v2` (top-level files, checkpoints not needed) is required to run
the inference web app.
