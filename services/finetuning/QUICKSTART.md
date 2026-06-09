# Quick Start Guide - Gender Bias Detection with Token Classification

## 5-Minute Setup

### 1. Navigate to finetuning service
```bash
cd services/finetuning
```

### 2. Install dependencies
```bash
pip install pyyaml datasets scikit-learn
```

### 3. Generate synthetic training data (5-10 minutes)
```bash
python scripts/01_generate_data.py --config config/config.yaml --num-samples 50000
```

Expected output:
```
Loading sentences from services/synthesizer_v2/output/label...
  Loaded 24000 sentences from gb_attack.json
  Loaded 20000 sentences from non_gb_neutral.json
  ...

Generating 5000 synthetic paragraphs...
  Generating 2000 paragraphs with no bias...
  Generating 2000 paragraphs with 1 bias...
  Generating 750 paragraphs with 2 biases...
  Generating 250 paragraphs with 3+ biases...

Saved 4000 samples to services/finetuning/data/train.jsonl
Saved 500 samples to services/finetuning/data/validation.jsonl
Saved 500 samples to services/finetuning/data/test.jsonl
```

### 4. Train the model (30-60 minutes on GPU, 3-5 hours on CPU)
```bash
python scripts/02_train.py --config config/config.yaml
```

Expected output:
```
Loading tokenizer and model: xlm-roberta-base
Model loaded successfully!
Number of parameters: 121,996,032

Preparing datasets...
Loaded datasets:
  Train: 4000 samples
  Validation: 500 samples
  Test: 500 samples

Starting training...
Epoch 1/3: 100%|████████| 250/250 [12:34<00:00, ...]
Epoch 2/3: 100%|████████| 250/250 [12:31<00:00, ...]
Epoch 3/3: 100%|████████| 250/250 [12:29<00:00, ...]

Test Results:
  eval_precision: 0.8923
  eval_recall: 0.8645
  eval_f1: 0.8782

Model saved to services/finetuning/models
```

### 5. Test the model
```bash
# Single text
python scripts/03_inference.py \
    --model-dir models \
    --text "ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีกวะ? งานวิจัยใหม่นี้จะช่วยแก้ปัญหาฝุ่น PM 2.5."

# Expected output:
# Bias Detection Results:
# Total sentences: 2
# Biased sentences: 1
# Bias percentage: 50.0%
#
# Biased Sentences:
#   [0] (95.23%) ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีกวะ?
```

## Understanding the Output

### Confidence Score
- **0.0-0.4**: Low confidence, likely not biased
- **0.4-0.7**: Medium confidence, some bias indicators
- **0.7-1.0**: High confidence, strong bias indicators

### Bias Tags Explained
- **B-BIAS**: Beginning of a biased phrase
- **I-BIAS**: Inside a biased phrase (continuation)
- **O**: Outside bias (neutral text)

Example:
```
Text: "ผู้หญิง [B-BIAS] สมัยนี้ [I-BIAS] นอกจาก [O] สวย..."
```

## Key Configuration Options

Edit `config/config.yaml` to change:

```yaml
# How many sentences to put in each synthetic paragraph
paragraph:
  min_sentences: 4
  max_sentences: 8

# What percentage of paragraphs should have bias
distribution:
  no_bias_ratio: 0.40        # 40% no bias
  one_bias_ratio: 0.40       # 40% 1 biased sentence
  two_bias_ratio: 0.15       # 15% 2 biased sentences
  three_plus_bias_ratio: 0.05 # 5% 3+ biased sentences

# Training settings
training:
  num_epochs: 3              # More epochs = longer training but potentially better
  batch_size: 16             # Reduce if out of memory
  learning_rate: 2e-5        # Don't change unless you know what you're doing
```

## Common Tasks

### Run on GPU instead of CPU
Edit `config/config.yaml`:
```yaml
model:
  device: "cuda"  # Changed from "cpu"
```

### Generate more training data
```bash
python scripts/01_generate_data.py --num-samples 20000
```

### Use different base model (smaller/faster)
Edit `config/config.yaml`:
```yaml
model:
  base_model: "bert-base-multilingual-cased"  # Smaller, faster
```

### Save model to custom location
The model automatically saves to `models/` after training. To save elsewhere:
```bash
# In Python script, change:
trainer.save_model('/path/to/custom/location')
```

### Batch process multiple texts
```python
from services.finetuning.src.inference import BiasDetector

detector = BiasDetector('models')

texts = [
    "First text here...",
    "Second text here...",
    "Third text here..."
]

results = detector.batch_detect_bias(texts)

for i, result in enumerate(results):
    print(f"Text {i}: {result['summary']['biased_count']} biased sentences")
```

## Troubleshooting

### Error: "Dataset files not found!"
**Solution**: Run step 3 (generate data) first

### Error: "Out of memory"
**Solution**: 
- Reduce `batch_size` in config.yaml (try 8 or 4)
- Reduce `max_length` to 256 or 384
- Use smaller model: `bert-base-multilingual-cased`

### Error: "Model not found"
**Solution**: Make sure training completed successfully (step 4)

### Training is very slow
**Solution**:
- Use GPU: Set `device: "cuda"` in config.yaml
- Reduce number of samples: `--num-samples 5000`
- Reduce epochs: Change `num_epochs: 1` temporarily to test

## Next Steps After Training

1. **Evaluate performance**: Check precision/recall on test set
2. **Fine-tune hyperparameters**: Try different learning rates, epochs
3. **Add to web app**: Integrate with Flask/FastAPI
4. **Test on real data**: Use actual paragraphs from your domain
5. **Improve dataset**: Add more diverse bias examples

## Structure of Generated Data

When you run step 3 (generate data), each example looks like:

```json
{
  "text": "สมองกลวงปะ? งานวิจัยใหม่จะช่วยแก้ปัญหา.",
  "sentences": ["สมองกลวงปะ?", "งานวิจัยใหม่จะช่วยแก้ปัญหา."],
  "token_labels": [1, 1, 0, 0, 0, 0, 0, 0],
  "sentence_labels": [1, 0],
  "bias_info": [
    {
      "text": "สมองกลวงปะ?",
      "index": 0,
      "subtype": "GB-ATTACK",
      "target": "ผู้หญิง"
    }
  ]
}
```

Where:
- **token_labels**: 1 = token is biased, 0 = token is not biased
- **sentence_labels**: 1 = sentence contains bias, 0 = sentence is neutral
- **bias_info**: Details about which sentences are biased and why

## Memory Requirements

| Model | VRAM Needed | RAM Needed | Speed |
|-------|-------------|-----------|-------|
| bert-base-multilingual-cased | 2GB | 8GB | ~2-3 min/epoch |
| xlm-roberta-base | 3GB | 12GB | ~3-4 min/epoch |
| xlm-roberta-large | 6GB | 16GB | ~6-8 min/epoch |

Note: Requirements assume batch_size=16 and max_length=512

## Need Help?

1. Check the main README.md in this directory
2. Review the configuration file: config/config.yaml
3. Check model-specific docs: https://huggingface.co/xlm-roberta-base
4. Token classification tutorial: https://huggingface.co/docs/transformers/tasks/token_classification
