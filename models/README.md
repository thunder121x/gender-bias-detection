# Models

Model weights are **not stored in git**. Both fine-tuned models are published on
Hugging Face:

| Model | Hugging Face |
|-------|--------------|
| MiniLM gender-bias classifier (`minilm_gender_bias_v2`) | [thunder121x/thai-gender-bias-classifier-minilm](https://huggingface.co/thunder121x/thai-gender-bias-classifier-minilm) |
| Qwen 3.5 2B span-extraction LoRA (`gender_bias_qwen35_2b_unsloth`) | [thunder121x/thai-gender-bias-span-extraction-qwen3.5-2b](https://huggingface.co/thunder121x/thai-gender-bias-span-extraction-qwen3.5-2b) |

## Option 1 — load directly from the Hub (recommended)

`transformers` accepts the Hub id anywhere a local path is used:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "thunder121x/thai-gender-bias-classifier-minilm"
)
tokenizer = AutoTokenizer.from_pretrained(
    "thunder121x/thai-gender-bias-classifier-minilm"
)
```

For the inference web app, point `MODEL_PATH` at the Hub id:

```bash
MODEL_PATH=thunder121x/thai-gender-bias-classifier-minilm \
    python services/inference_app/app.py
```

## Option 2 — download into this folder

```bash
pip install -U "huggingface_hub[cli]"
hf download thunder121x/thai-gender-bias-classifier-minilm \
    --local-dir models/minilm_gender_bias_v2
hf download thunder121x/thai-gender-bias-span-extraction-qwen3.5-2b \
    --local-dir models/gender_bias_qwen35_2b_unsloth
```

Expected layout:

```
models/
├── minilm_gender_bias_v2/          # classifier — used by services/inference_app
└── gender_bias_qwen35_2b_unsloth/  # LoRA adapter — used by services/lora_finetuning
```

## Related Hugging Face repos

- Base model for retraining: [repelloai/CREST-Base](https://huggingface.co/repelloai/CREST-Base)
  (the local `crest_base_local/` copy was just a cache of this)
- Training data: [thunder121x/thai-gender-bias-training-data](https://huggingface.co/datasets/thunder121x/thai-gender-bias-training-data)
