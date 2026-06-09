# Thai Gender Bias Detection

Detection of gender bias in Thai social-media text. End-to-end pipeline: scrape
social platforms → cluster topics → annotate → synthesize training data → fine-tune
classifiers → analyze errors → visualize → serve an inference web app.

Senior project, MIT licensed.

## Pipeline

```
scrape (YouTube/TikTok/Twitter)        services/scraper
        │
cluster topics (BERTopic, Thai)        services/clustering, services/visualization
        │
annotate (guideline + web tool)        services/annotator, docs/annotation-guideline.md
        │
synthesize training data (Gemini)      services/synthesizer, services/prompteng
        │
fine-tune classifiers                  services/finetuning (MiniLM), services/lora_finetuning (Qwen LoRA)
        │
auto-analyze errors                    services/auto_analysis
        │
inference web app (Flask)              services/inference_app
```

## Repository structure

| Path | What it is |
|------|------------|
| [services/scraper/](services/scraper/) | YouTube / TikTok / Twitter comment scrapers (Playwright, Selenium, API) |
| [services/clustering/](services/clustering/) | BERTopic topic-modeling pipeline for Thai text |
| [services/visualization/](services/visualization/) | BERTopic visualization notebooks |
| [services/annotator/](services/annotator/) | Web-based annotation tool (Vite + Tailwind) |
| [services/synthesizer/](services/synthesizer/) | Gemini-based synthetic training-data generator |
| [services/prompteng/](services/prompteng/) | Prompt-engineering experiments |
| [services/finetuning/](services/finetuning/) | MiniLM classifier fine-tuning (H100 scripts in `scripts/`) |
| [services/lora_finetuning/](services/lora_finetuning/) | Qwen 3.5 2B LoRA span detector (Unsloth) |
| [services/auto_analysis/](services/auto_analysis/) | Automated error analysis with Gemini validation |
| [services/inference_app/](services/inference_app/) | Flask web app serving the trained classifier |
| [data/](data/) | Annotated datasets (CSV, tracked in git) |
| [notebooks/](notebooks/) | Exploration & training notebooks |
| [docs/](docs/) | Annotation guideline, data format spec, fine-tuning guides |
| [report/](report/) | Senior project report (LaTeX source + compiled PDF) |
| [models/](models/) | Model weights — download from Drive, see [models/README.md](models/README.md) |

## Data & Models

Large artifacts (~8.3GB: model weights, raw scraped data, training data, outputs)
are stored on Google Drive, not in git:

**Drive folder:** `<DRIVE_LINK>` <!-- paste share link after upload -->

| Drive path | Contents |
|------------|----------|
| `models/` | MiniLM classifier, CREST base, Qwen LoRA adapter |
| `training_data/` | LoRA fine-tuning train/val JSONL |
| `outputs/` | Synthesis outputs, visualization assets |
| `scraper_data/` | Raw scraped tweets, post-processed clustering input |

Small annotated datasets (~3MB) are tracked in [data/](data/).

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/thunder121x/gender-bias-detection.git
cd gender-bias-detection
cp .env.example .env        # fill in GEMINI_API_KEY etc.
```

Each service is self-contained — install its own dependencies:

```bash
# example: run the inference web app
pip install -r services/inference_app/requirements.txt
# download models/minilm_gender_bias_v2 from Drive first (see models/README.md)
python services/inference_app/app.py
```

Services with a `pyproject.toml` (scraper, clustering, prompteng, synthesizer) can be
installed with `pip install -e services/<name>` or [uv](https://docs.astral.sh/uv/).

## Documentation

- [Annotation guideline](docs/annotation-guideline.md) — label definitions, decision rules (Thai)
- [Data format specification](docs/data-format-specification.md) — dataset schemas
- [Fine-tuning guide](docs/finetuning-guide.md) — task definition, training protocol, evaluation
- [H100 training guide](docs/h100-training-guide.md) — GPU setup, presets, troubleshooting
- [Synthesis prompt example](docs/example-prompt-synthesize.md)

## License

[MIT](LICENSE)
