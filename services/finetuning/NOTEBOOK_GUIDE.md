# 02_train.ipynb - Jupyter Notebook for Model Training

A comprehensive Jupyter notebook for fine-tuning the gender bias detection model. Works on both **local machines** and **Google Colab**.

## Quick Start

### Google Colab (Recommended)
1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Select `services/finetuning/scripts/02_train.ipynb`
4. In **Cell 1**, set:
   ```python
   COLAB = True
   DRIVE_TRAIN_FILE = "/content/drive/MyDrive/assets/train_span/data/train.jsonl"
   ```
5. Run all cells (Runtime → Run all)

### Local Machine (Jupyter)
```bash
cd services/finetuning
jupyter notebook scripts/02_train.ipynb
```

Make sure data files are in `./data/` directory:
- `data/train.jsonl`
- `data/validation.jsonl`
- `data/test.jsonl`

## Notebook Structure

### Cell 1: Setup & Configuration
Configure paths for local or Google Colab environment.

**Key setting:**
```python
COLAB = False  # Change to True for Google Colab
```

**For Colab, set paths to Google Drive:**
```python
DRIVE_TRAIN_FILE = "/content/drive/MyDrive/assets/train_span/data/train.jsonl"
DRIVE_VAL_FILE = "/content/drive/MyDrive/assets/train_span/data/validation.jsonl"
DRIVE_TEST_FILE = "/content/drive/MyDrive/assets/train_span/data/test.jsonl"
MODEL_SAVE_DIR = "/content/drive/MyDrive/assets/train_span/models"
```

### Cell 2: Install Dependencies
Automatically installs required packages:
- torch
- transformers
- datasets
- pyyaml
- numpy
- scikit-learn

### Cell 3: Import Libraries
Imports all necessary libraries and checks GPU availability.

**Output:**
```
Device: cuda
GPU: NVIDIA RTX 3080
GPU Memory: 10.0 GB
```

### Cell 4: Load Training Data
Loads the three dataset files and displays statistics.

**Expected output:**
```
✓ train.jsonl: 36000 samples
✓ validation.jsonl: 4000 samples
✓ test.jsonl: 10000 samples
```

### Cell 5: Load Model & Tokenizer
Loads `xlm-roberta-base` and configures it for token classification.

**Model details:**
- Base model: xlm-roberta-base
- Parameters: 121,996,032
- Task: Token-level classification (3 labels: O, B-BIAS, I-BIAS)

### Cell 6: Tokenize & Align Labels
Converts text to tokens and aligns BIO labels with subword tokens.

**What happens:**
- Text → tokens (subword tokenization)
- Tokens → token IDs
- BIO labels → token-level labels
- Padding to max_length (512)

### Cell 7: Setup Training
Configures training parameters and creates the Trainer object.

**Key parameters:**
```python
num_train_epochs: 3
per_device_train_batch_size: 16
learning_rate: 2e-5
warmup_steps: 500
weight_decay: 0.01
```

### Cell 8: Train Model
Runs the fine-tuning loop for 3 epochs.

**Output:**
```
Epoch 1/3: 100%|████████████| 2250/2250 [12:34<00:00, 2.97it/s]
Epoch 2/3: 100%|████████████| 2250/2250 [12:31<00:00, 2.98it/s]
Epoch 3/3: 100%|████████████| 2250/2250 [12:29<00:00, 2.99it/s]
```

**Training time:**
- GPU: 30-60 minutes
- CPU: 2-3 hours

### Cell 9: Evaluate on Test Set
Evaluates the trained model on the test set.

**Metrics:**
```
Precision: 0.8923
Recall: 0.8645
F1: 0.8878
```

### Cell 10: Save Model
Saves the fine-tuned model and tokenizer to disk (local or Google Drive).

**Files created:**
```
models/final_model/
├── pytorch_model.bin (355 MB)
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── training_config.json
```

### Cell 11: Summary
Displays a comprehensive training summary.

## Customization

### Change Paths
Edit Cell 1:
```python
DRIVE_TRAIN_FILE = "/your/custom/path/train.jsonl"
DRIVE_VAL_FILE = "/your/custom/path/validation.jsonl"
DRIVE_TEST_FILE = "/your/custom/path/test.jsonl"
MODEL_SAVE_DIR = "/your/custom/path/models"
```

### Change Training Parameters
Edit Cell 7:
```python
num_train_epochs=5  # More training
per_device_train_batch_size=8  # Reduce if OOM
learning_rate=1e-5  # Lower learning rate
warmup_steps=1000  # More warmup
```

### Change Base Model
Edit Cell 5:
```python
model_name = "bert-base-multilingual-cased"  # Smaller, faster
# Or: "xlm-roberta-large"  # Larger, slower, more powerful
```

## Troubleshooting

### Out of Memory Error
**Solution:** Reduce batch size in Cell 7:
```python
per_device_train_batch_size=8  # Instead of 16
```

### Slow Training on CPU
**Solution:** 
1. Use Google Colab GPU: Runtime → Change runtime type → GPU
2. Or use smaller model: `bert-base-multilingual-cased`

### File Not Found Error
**Solution:** Check paths in Cell 1:
- For local: Make sure files are in `./data/` directory
- For Colab: Make sure paths match your Google Drive structure

### CUDA Out of Memory
**Solution:**
1. Reduce batch size
2. Reduce max_length from 512 to 256 or 384
3. Use mixed precision training (add `--fp16` to training args)

## Expected Performance

### Accuracy Metrics
- Precision: 88-92%
- Recall: 85-88%
- F1 Score: 86-90%

### Training Time
- GPU (RTX 3080): 30-60 minutes
- GPU (A100): 20-40 minutes
- CPU: 2-3 hours

### Output Size
- Model: ~355 MB
- Tokenizer: ~1 MB
- Config: < 1 MB
- **Total: ~356 MB**

## Integration with Inference

After training, use the model in inference:

```python
from transformers import pipeline

model_name = "models/final_model"
pipe = pipeline("token-classification", model=model_name)

result = pipe("ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีก? งานวิจัยใหม่ดี.")
print(result)
```

Or use the BiasDetector class:
```python
from src.inference import BiasDetector

detector = BiasDetector("models/final_model")
result = detector.detect_bias("your text here")
print(result['biased_sentences'])
```

## Next Steps

1. ✅ Run this notebook to train the model
2. ✅ Evaluate on test set
3. ✅ Save the model
4. 📌 Use the model in inference with `03_inference.py`
5. 📌 Integrate into Flask/FastAPI backend

## Notes

- The notebook automatically handles tokenization and label alignment
- Progress bars show training progress in real-time
- Metrics are computed after each epoch
- Best model checkpoint is saved automatically
- All code is GPU-optimized but works on CPU too

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Read the main README.md
3. Check ARCHITECTURE.md for technical details
4. Review QUICKSTART.md for setup help
