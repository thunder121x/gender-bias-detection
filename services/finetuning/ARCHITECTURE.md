# Token Classification Architecture & Implementation Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input: Paragraph Text                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Sentence Splitting (by ". ")                        │
│  "sentence1. sentence2. sentence3" → ["sent1", "sent2", "sent3"]│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         HuggingFace Tokenizer (xlm-roberta-base)                │
│  ["sent1", "sent2"] → tokens ["ผู้", "หญิง", "สม", "ัย", ...]  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        Fine-tuned Token Classification Model                     │
│                  (3 labels: O, B-BIAS, I-BIAS)                  │
│  Input: Token IDs → Output: Label logits for each token         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Post-processing & Aggregation                       │
│  - Merge token labels into spans                                │
│  - Map back to sentences                                        │
│  - Calculate confidence scores                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Output: Biased Sentences                        │
│  {                                                               │
│    "sentences": ["sent1", "sent2", "sent3"],                    │
│    "biased_sentences": [                                        │
│      {                                                          │
│        "text": "sent1",                                         │
│        "index": 0,                                              │
│        "confidence": 0.95,                                      │
│        "bias_spans": [{"text": "ผู้หญิง", "start": 0, ...}]   │
│      }                                                          │
│    ]                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Training Pipeline

### Phase 1: Data Generation

```python
DataAugmenter
├─ Load sentences from synthesizer_v2/output/label/
│  ├─ GB files: gb_attack.json, gb_normative.json, gb_sex.json
│  └─ NON-GB files: non_gb_neutral.json, non_gb_insult.json, non_gb_meta.json
│
├─ Generate synthetic paragraphs
│  ├─ Sample 2-8 sentences per paragraph
│  ├─ Mix GB and NON-GB sentences randomly
│  ├─ Maintain bias distribution (40% no-bias, 40% 1-bias, 15% 2-bias, 5% 3+-bias)
│  └─ Create BIO labels for each token
│
└─ Output: train.jsonl, validation.jsonl, test.jsonl
```

### Phase 2: Model Fine-tuning

```python
TokenClassificationTrainer
├─ Load base model: xlm-roberta-base (121M parameters)
├─ Load tokenizer: AutoTokenizer.from_pretrained()
├─ Add classification head (3 labels)
│
├─ Training loop
│  ├─ Batch size: 16
│  ├─ Learning rate: 2e-5
│  ├─ Epochs: 3
│  ├─ Warmup steps: 500
│  ├─ Loss: Cross-entropy for token classification
│  └─ Optimizer: AdamW
│
├─ Validation after each epoch
└─ Save best checkpoint
```

### Phase 3: Inference

```python
BiasDetector
├─ Load fine-tuned model and tokenizer
├─ For each input paragraph:
│  ├─ Split into sentences
│  ├─ For each sentence:
│  │  ├─ Tokenize
│  │  ├─ Run through model
│  │  ├─ Get logits for each token
│  │  ├─ Apply softmax to get probabilities
│  │  ├─ Extract bias spans (consecutive B-BIAS/I-BIAS tokens)
│  │  └─ Calculate sentence confidence (max probability)
│  └─ Aggregate results
└─ Return: List of biased sentences with positions and confidence
```

## Key Algorithms

### 1. BIO Tag Assignment

When creating training data, tokens are labeled with BIO tags based on sentence bias status:

```python
def create_token_labels(sentence_texts, sentence_labels):
    token_labels = []
    first_token_in_sentence = True
    
    for sentence, label in zip(sentence_texts, sentence_labels):
        tokens = sentence.split()
        for token in tokens:
            if label == 1:  # Biased sentence
                if first_token_in_sentence:
                    token_labels.append(1)  # B-BIAS
                    first_token_in_sentence = False
                else:
                    token_labels.append(2)  # I-BIAS
            else:  # Non-biased
                token_labels.append(0)  # O
        first_token_in_sentence = True
    
    return token_labels
```

### 2. Token-Sentence Alignment

Subword tokenization (e.g., "ผู้หญิง" → ["ผู้", "หญิง"]) requires alignment:

```python
def align_labels_with_tokens(word_ids, sentence_labels):
    labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:  # Special tokens
            labels.append(-100)  # Ignored in loss
        elif word_idx != previous_word_idx:  # First token of word
            labels.append(sentence_labels[word_idx])
        else:  # Continuation token
            labels.append(sentence_labels[word_idx])
        
        previous_word_idx = word_idx
    
    return labels
```

### 3. Confidence Calculation

```python
def get_sentence_confidence(token_logits, token_labels):
    """
    Calculate sentence-level confidence based on token probabilities.
    Returns the maximum probability among BIAS tokens in the sentence.
    """
    probs = softmax(token_logits, dim=-1)
    bias_token_probs = []
    
    for prob, label in zip(probs, token_labels):
        if label in [1, 2]:  # B-BIAS or I-BIAS
            bias_prob = prob[label].item()
            bias_token_probs.append(bias_prob)
    
    return max(bias_token_probs) if bias_token_probs else 0.0
```

## Module Dependencies

```
services/finetuning/
│
├── config.yaml (configuration)
│
├── scripts/
│   ├── 01_generate_data.py
│   │   └── imports: data_augmenter.DataAugmenter
│   │   └── outputs: train.jsonl, validation.jsonl, test.jsonl
│   │
│   ├── 02_train.py
│   │   └── imports: trainer.TokenClassificationTrainer
│   │   └── inputs: train.jsonl, validation.jsonl
│   │   └── outputs: models/checkpoint-*
│   │
│   └── 03_inference.py
│       └── imports: inference.BiasDetector
│       └── inputs: models/checkpoint-*
│
└── src/
    ├── data_augmenter.py
    │   ├── class: DataAugmenter
    │   ├── methods:
    │   │   ├─ load_sentences()
    │   │   ├─ generate_dataset()
    │   │   ├─ _generate_paragraph()
    │   │   ├─ _create_token_labels()
    │   │   └─ save_dataset()
    │   └── dependencies: json, random, os
    │
    ├── dataset_processor.py
    │   ├── class: TokenClassificationDataset
    │   ├── methods:
    │   │   ├─ tokenize_and_align_labels()
    │   │   ├─ align_labels_with_tokens()
    │   │   └─ prepare_dataset()
    │   └── dependencies: transformers.AutoTokenizer
    │
    ├── trainer.py
    │   ├── class: TokenClassificationTrainer
    │   ├── methods:
    │   │   ├─ setup()
    │   │   ├─ load_and_prepare_datasets()
    │   │   ├─ train()
    │   │   ├─ evaluate()
    │   │   └─ save_model()
    │   └── dependencies: transformers, datasets, torch
    │
    └── inference.py
        ├── class: BiasDetector
        ├── methods:
        │   ├─ detect_bias()
        │   ├─ batch_detect_bias()
        │   ├─ _predict_sentence()
        │   ├─ _extract_bias_spans()
        │   └─ highlight_text()
        └── dependencies: transformers, torch
```

## Data Format Specifications

### Input: Synthetic Paragraph (JSONL)
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
      "target": "gender_group"
    }
  ]
}
```

### Model Input (Tokenized)
```python
{
  "input_ids": [101, 2803, 2544, ...],  # Token IDs
  "attention_mask": [1, 1, 1, ...],     # Attention mask
  "token_type_ids": [0, 0, 0, ...],     # Token type (BERT)
  "labels": [0, 0, 1, 1, 0, ...]        # Label IDs for loss
}
```

### Model Output (Logits)
```python
{
  "logits": shape (batch_size, seq_length, num_labels)  # 3 labels
  # For each token: [score_O, score_B-BIAS, score_I-BIAS]
}
```

### Inference Output: Biased Sentences
```json
{
  "paragraph": "input text",
  "sentences": ["sent1", "sent2", ...],
  "biased_sentences": [
    {
      "text": "sentence with bias",
      "index": 0,
      "confidence": 0.95,
      "tokens": [
        {"token": "ผู้", "label": "B-BIAS", "confidence": 0.95},
        {"token": "หญิง", "label": "I-BIAS", "confidence": 0.92}
      ],
      "bias_spans": [
        {
          "text": "ผู้หญิง",
          "start": 0,
          "end": 1
        }
      ]
    }
  ],
  "summary": {
    "total_sentences": 3,
    "biased_count": 1,
    "bias_percentage": 33.3
  }
}
```

## Performance Characteristics

### Training Time (per epoch)
- GPU (NVIDIA RTX 3080): ~3-4 minutes per epoch
- GPU (NVIDIA A100): ~2-3 minutes per epoch  
- CPU: ~30-45 minutes per epoch

### Memory Usage
- Model size: 355MB (xlm-roberta-base)
- VRAM required: ~3GB (batch_size=16, max_length=512)
- RAM required: ~12GB

### Inference Speed
- Single sentence: ~50-100ms (GPU), ~500-1000ms (CPU)
- Batch of 100 sentences: ~100-150ms per sentence (GPU)

## Model Selection Guide

| Scenario | Recommended Model | Trade-off |
|----------|-------------------|-----------|
| Maximum accuracy | xlm-roberta-large | Slower, more memory |
| Balanced | xlm-roberta-base | Default choice |
| Speed critical | bert-base-multilingual-cased | Slightly lower accuracy |
| Mobile/Edge | distilbert-base-multilingual-cased | Fastest, less accurate |

## Hyperparameter Tuning Guide

### Learning Rate
- **Too high (>5e-5)**: Training unstable, loss spikes
- **Too low (<1e-6)**: Training very slow, underfitting
- **Optimal (2e-5)**: Smooth convergence

### Batch Size
- **Too small (2-4)**: Noisy gradients, slower convergence
- **Too large (32-64)**: Less frequent updates, underfitting
- **Optimal (16)**: Balanced

### Number of Epochs
- **1 epoch**: Underfitting, 70-75% accuracy
- **3 epochs**: Good convergence, 85-90% accuracy
- **5+ epochs**: Diminishing returns, risk of overfitting

### Warmup Steps
- **Too small (<500)**: Unstable early training
- **Optimal (500-1000)**: Smooth early learning
- **Too large (>5000)**: Wastes training steps

## Debugging & Monitoring

### Check training progress:
```bash
tensorboard --logdir services/finetuning/logs
```

### Common issues:
1. **Loss not decreasing**: Reduce learning rate or increase warmup steps
2. **Training too slow**: Use GPU, reduce dataset size, reduce max_length
3. **High validation loss**: Increase epochs, adjust learning rate, add regularization
4. **Inference crashes**: Check VRAM, reduce batch size or max_length

## Future Improvements

1. **Ensemble models**: Combine multiple fine-tuned models
2. **Domain-specific training**: Fine-tune on real-world paragraphs
3. **Span-level detection**: Detect exact bias phrases instead of sentences
4. **Multi-label classification**: Support multiple bias types per sentence
5. **Attention visualization**: Show which tokens influence predictions
6. **Active learning**: Iteratively improve with human feedback
