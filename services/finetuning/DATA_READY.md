# ✅ Training Data Ready!

## Generation Summary

Successfully generated **50,000 synthetic paragraphs** for token-level gender bias detection!

### Dataset Breakdown
- **Total paragraphs**: 50,000
- **Paragraphs with bias**: 30,000 (60.0%)
- **Total biased sentences**: 45,005
- **Average biased sentences per biased paragraph**: 1.50

### Data Split
- **Training**: 36,000 samples (72%)
- **Validation**: 4,000 samples (8%)
- **Testing**: 10,000 samples (20%)

### Data Distribution
- **No bias**: 20,000 paragraphs (40%)
- **1 biased sentence**: 20,000 paragraphs (40%)
- **2 biased sentences**: 7,500 paragraphs (15%)
- **3+ biased sentences**: 2,500 paragraphs (5%)

## Data Format

Each paragraph includes:
- **text**: Full paragraph (space-separated sentences)
- **sentences**: List of individual sentences
- **sentence_labels**: [0=non-biased, 1=biased] for each sentence
- **token_labels**: BIO labels for each word token
- **bias_info**: Details about which sentences are biased
- **num_sentences**: Total number of sentences in paragraph

### Example
```json
{
  "text": "sentence1. sentence2. sentence3.",
  "sentences": ["sentence1", "sentence2", "sentence3"],
  "sentence_labels": [0, 0, 1],
  "token_labels": [0, 0, 1, 1, 0, ...],
  "bias_info": [
    {
      "text": "sentence3",
      "index": 2,
      "subtype": "GB-ATTACK",
      "target": "gender_group"
    }
  ],
  "num_sentences": 3
}
```

## Files Generated

```
services/finetuning/data/
├── train.jsonl        (36,000 lines)
├── validation.jsonl   (4,000 lines)
└── test.jsonl         (10,000 lines)
```

## Next Steps

### Step 1: Train the Model
```bash
cd services/finetuning
python scripts/02_train.py
```

Expected:
- Training time: 30-60 minutes on GPU
- Model: xlm-roberta-base (121M parameters)
- Accuracy: 85-90% on test set
- Output: Saved model in `models/` directory

### Step 2: Test Inference
```bash
python scripts/03_inference.py \
    --model-dir models \
    --text "ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีก? งานวิจัยใหม่ดี."
```

## Data Statistics

### By Bias Type
- **GB-ATTACK**: ~35% of all biased sentences
- **GB-SEX**: ~35% of all biased sentences
- **GB-NORMATIVE**: ~30% of all biased sentences

### By Paragraph Length
- **4 sentences**: ~25%
- **5 sentences**: ~25%
- **6 sentences**: ~25%
- **7-8 sentences**: ~25%

### Sentence Position in Biased Paragraphs
- First sentence: ~40% (higher chance of bias)
- Middle sentences: ~35%
- Last sentence: ~25%

## Quality Metrics

✅ **Balanced distribution**: 60% biased, 40% non-biased
✅ **Diverse sources**: Mix of 3 GB types × 3 NON-GB types
✅ **Complete labels**: BIO tags for all tokens
✅ **Metadata preserved**: Bias type and target included

## Configuration Used

```yaml
# Paragraph composition
min_sentences: 4
max_sentences: 8

# Bias distribution
no_bias_ratio: 0.40
one_bias_ratio: 0.40
two_bias_ratio: 0.15
three_plus_bias_ratio: 0.05

# Training split
train_test_split: 0.80
validation_split: 0.10
```

## Ready for Training!

Your data is now ready for fine-tuning. Run:
```bash
python scripts/02_train.py
```

Estimated training time:
- GPU: 30-60 minutes
- CPU: 2-3 hours

Good luck! 🚀
