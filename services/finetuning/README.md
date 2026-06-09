# Gender Bias Token Classification Fine-tuning Service

This service implements a **sentence-level gender bias detection** system using token classification (BIO tagging). Instead of classifying entire texts, it identifies which specific sentences contain gender bias within paragraphs.

## Overview

### Problem
Previous classification approach: Given a sentence → Label as GB or NON-GB
Current approach: Given a paragraph → Return list of biased sentences with their positions

### Solution
Token-level classification with BIO (Begin-Inside-Outside) tags:
- **B-BIAS**: Beginning of a biased span
- **I-BIAS**: Inside a biased span
- **O**: Outside (non-biased)

### Architecture

```
Input Paragraph
       ↓
[Sentence Splitter]
       ↓
[Fine-tuned Tokenizer + Model]
       ↓
[Token Classification (BIO)]
       ↓
[Sentence-level Aggregation]
       ↓
Output: List of biased sentences with positions
```

## Directory Structure

```
services/finetuning/
├── config/
│   └── config.yaml                 # Configuration file
├── src/
│   ├── __init__.py
│   ├── data_augmenter.py           # Synthetic data generation
│   ├── dataset_processor.py        # Dataset formatting for token classification
│   ├── trainer.py                  # Model fine-tuning
│   └── inference.py                # Inference pipeline
├── scripts/
│   ├── 01_generate_data.py         # Generate synthetic paragraphs
│   ├── 02_train.py                 # Train the model
│   └── 03_inference.py             # Run inference
├── models/                          # Saved models
├── data/                            # Generated datasets
└── logs/                            # Training logs
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.1.1+
- Transformers 4.35.2+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
cd services/finetuning

# Install additional dependencies
pip install pyyaml datasets scikit-learn
```

## Usage

### Step 1: Generate Synthetic Training Data

The data augmentation pipeline creates synthetic paragraphs by combining:
- **GB sentences**: From `services/synthesizer_v2/output/label/{gb_attack,gb_sex,gb_normative}.json`
- **NON-GB sentences**: From `services/synthesizer_v2/output/label/{non_gb_*.json}`

Generate synthetic paragraphs:
```bash
python scripts/01_generate_data.py \
    --config config/config.yaml \
    --num-samples 10000
```

This creates:
- `data/train.jsonl` (80% of data, 72% used for training, 10% used for validation)
- `data/validation.jsonl` (10% of data, 10% used for validation)
- `data/test.jsonl` (10% of data, for testing)

**Data Format:**
```json
{
  "text": "sentence1. sentence2. sentence3.",
  "sentences": ["sentence1", "sentence2", "sentence3"],
  "token_labels": [0, 0, 1, 1, 0, ...],
  "sentence_labels": [0, 0, 1],
  "bias_info": [
    {
      "text": "sentence3",
      "index": 2,
      "subtype": "GB-ATTACK",
      "target": "ผู้หญิง"
    }
  ]
}
```

### Step 2: Fine-tune Model

Train the model on the generated data:
```bash
python scripts/02_train.py --config config/config.yaml
```

The script will:
1. Load `xlm-roberta-base` (Thai-compatible multilingual BERT)
2. Fine-tune for token classification (3 epochs)
3. Evaluate on validation set
4. Save best model to `models/`

**Key Parameters:**
- `batch_size`: 16
- `learning_rate`: 2e-5
- `num_epochs`: 3
- `max_length`: 512 tokens

### Step 3: Run Inference

#### Single text:
```bash
python scripts/03_inference.py \
    --model-dir models/checkpoint-latest \
    --text "ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีกวะ? คนดีๆ ไม่เหลือเลย."
```

#### With highlight output:
```bash
python scripts/03_inference.py \
    --model-dir models/checkpoint-latest \
    --text "ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีกวะ? คนดีๆ ไม่เหลือเลย." \
    --highlight
```

#### JSON output:
```bash
python scripts/03_inference.py \
    --model-dir models/checkpoint-latest \
    --text "your text here" \
    --json
```

#### Interactive mode (read from stdin):
```bash
python scripts/03_inference.py --model-dir models/checkpoint-latest
# Then paste your text and press Ctrl+D
```

## Output Format

### Standard Output
```
Bias Detection Results:
Total sentences: 3
Biased sentences: 2
Bias percentage: 66.7%

Biased Sentences:
  [0] (95.23%) ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีกวะ?
       -> ผู้หญิงสมัยนี้นอกจากสวยแล้ว
  [2] (87.15%) ผู้ชายหัวหนี.
```

### JSON Output
```json
{
  "paragraph": "...",
  "sentences": ["...", "...", "..."],
  "biased_sentences": [
    {
      "text": "...",
      "index": 0,
      "confidence": 0.9523,
      "tokens": [
        {"token": "ผู้", "label": "B-BIAS", "confidence": 0.95},
        {"token": "หญิง", "label": "I-BIAS", "confidence": 0.92},
        ...
      ],
      "bias_spans": [
        {
          "text": "ผู้หญิงสมัยนี้",
          "start": 0,
          "end": 3
        }
      ]
    }
  ],
  "summary": {
    "total_sentences": 3,
    "biased_count": 2,
    "bias_percentage": 66.7
  }
}
```

## Configuration

Edit `config/config.yaml` to customize:

### Data Generation
```yaml
data:
  paragraph:
    min_sentences: 4          # Minimum sentences per paragraph
    max_sentences: 8          # Maximum sentences per paragraph
    distribution:
      no_bias_ratio: 0.40     # % paragraphs with no bias
      one_bias_ratio: 0.40    # % paragraphs with 1 biased sentence
      two_bias_ratio: 0.15    # % paragraphs with 2 biased sentences
      three_plus_bias_ratio: 0.05
```

### Model
```yaml
model:
  base_model: "xlm-roberta-base"  # Can also use:
                                   # - bert-base-multilingual-cased
                                   # - xlm-roberta-large (slower, more powerful)
  max_length: 512                  # Max sequence length
  device: "cuda"                   # or "cpu"
```

### Training
```yaml
training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 500
  weight_decay: 0.01
```

## Model Comparison

| Model | Size | Speed | Quality | Thai | Recommended |
|-------|------|-------|---------|------|-------------|
| xlm-roberta-base | 355M | Fast | Good | Yes | ✓ |
| bert-base-multilingual-cased | 167M | Fastest | Fair | Yes | - |
| xlm-roberta-large | 550M | Slow | Excellent | Yes | - |

## API Integration

To use the bias detector in your application:

```python
from services.finetuning.src.inference import BiasDetector

# Initialize detector
detector = BiasDetector('services/finetuning/models/checkpoint-latest')

# Detect bias
result = detector.detect_bias(
    "ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีกวะ? สมองกลวงปะ?",
    confidence_threshold=0.5
)

# Access results
print(f"Biased sentences: {result['biased_sentences']}")
print(f"Summary: {result['summary']}")

# Get highlighted text
highlighted = detector.highlight_text(result)
print(highlighted)
```

## Training Workflow

```
1. Load sentences from synthesizer_v2/output/label/*.json
   ↓
2. Generate synthetic paragraphs (10000 samples)
   ├─ 40% no bias
   ├─ 40% 1 biased sentence
   ├─ 15% 2 biased sentences
   └─ 5% 3+ biased sentences
   ↓
3. Create token-level BIO labels
   ↓
4. Split into train/val/test
   ↓
5. Fine-tune xlm-roberta-base
   ├─ 3 epochs
   ├─ Batch size 16
   ├─ Learning rate 2e-5
   └─ Warmup 500 steps
   ↓
6. Evaluate on test set
   ↓
7. Save model checkpoint
```

## Metrics

### Token-level Metrics
- **Precision**: Of predicted bias tokens, how many are actually biased?
- **Recall**: Of actual bias tokens, how many did we find?
- **F1 Score**: Harmonic mean of precision and recall

### Sentence-level Metrics
- **Accuracy**: Percentage of sentences correctly classified
- **Precision/Recall**: At sentence level (all tokens in sentence biased = biased)

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `max_length`
- Use `bert-base-multilingual-cased` instead of `xlm-roberta-base`

### Slow Training
- Use GPU: Set `device: "cuda"` in config
- Reduce `num_epochs`
- Reduce number of samples in generation

### Low Accuracy
- Increase `num_samples` in data generation
- Adjust `distribution` to match your data
- Train for more epochs: increase `num_epochs`

## Next Steps

1. **Evaluate on real data**: Test on actual Thai text
2. **Fine-tune hyperparameters**: Adjust learning rate, batch size, epochs
3. **Add more data sources**: Combine with other bias datasets
4. **Deploy as API**: Integrate with Flask/FastAPI for web service
5. **Span-level detection**: Fine-tune to extract exact bias phrases

## References

- HuggingFace Token Classification: https://huggingface.co/docs/transformers/tasks/token_classification
- XLM-RoBERTa: https://huggingface.co/xlm-roberta-base
- BIO Tagging: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

## License

Same as parent project

## Authors

- **Nanphat Tongsirisukool** — nanphatx@hotmail.com
- **Natcha Trairattanasak** — pnbookclub@gmail.com
