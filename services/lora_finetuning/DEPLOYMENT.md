# Deployment Checklist — LoRA Fine-tuning Service

**Service**: Thai Gender Bias Span Detector  
**Date**: April 22, 2026  
**Status**: ✅ READY FOR VM DEPLOYMENT  
**Hardware Target**: RTX 6000 Blackwell (96GB VRAM)

---

## 📦 Pre-Deployment Verification

- [x] **src/ folder removed** — Eliminated duplicate implementations
- [x] **scripts/ folder removed** — No Jupyter notebooks needed for production
- [x] **Empty directories cleaned** — Removed data/, logs/, models/ directories
- [x] **Redundant docs removed** — Kept only essential documentation
- [x] **All Python scripts syntax-checked** — No compilation errors
- [x] **Training data present** — train.jsonl (203M) + val.jsonl (11M) ready
- [x] **Requirements file valid** — All dependencies listed

---

## 📋 Directory Structure (Final)

```
services/lora_finetuning/
├── manage.py                          [ENTRY POINT - Interactive menu]
├── finetune_qwen_span_detector.py     [Data preparation]
├── finetune_qwen_lora.py              [LoRA fine-tuning (30-60 min)]
├── inference_qwen_span.py             [Inference engine]
├── validate_system_prompt.py          [System prompt validation]
├── requirements_lora.txt              [Python dependencies]
├── training_data/                     [Pre-generated training dataset]
│   ├── train.jsonl                    [22.8k samples, 203 MB]
│   └── val.jsonl                      [1.2k samples, 11 MB]
├── START_HERE.txt                     [Quick start guide]
├── SERVICE_README.md                  [Service overview & usage]
├── README.md                          [Additional documentation]
└── DEPLOYMENT.md                      [This file]
```

**Total Size**: ~220 MB (fits easily on any VM)

---

## 🚀 Quick Start on VM

### 1. Transfer to VM
```bash
# From local machine
scp -r services/lora_finetuning/ user@vm-ip:/path/to/services/

# Or copy entire repo:
# scp -r gender-bias-detection/ user@vm-ip:/path/to/projects/
```

### 2. Install Dependencies
```bash
cd services/lora_finetuning
pip install -r requirements_lora.txt
```

### 3. Run Training
```bash
# Option A: Interactive menu (recommended)
python3 manage.py
# Then select: 1 (validate) → 2 (prepare) → 3 (train)

# Option B: Direct command
python3 manage.py train
```

### 4. Expected Results
- **Training time**: 30-60 minutes on RTX 6000 Blackwell
- **Output location**: `qwen_gb_detector_lora/` (auto-created)
- **Checkpoint frequency**: Every epoch saved automatically
- **Validation metric**: Accuracy logged after each epoch

---

## ✅ Pre-Training Checks

Before running on VM, verify:

```bash
# Check system prompt consistency
python3 validate_system_prompt.py
# Expected: ✅ All files have matching system prompt

# Check training data
wc -l training_data/*.jsonl
# Expected: ~22800 train samples, ~1200 val samples

# Check Python path
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: True (if GPU drivers installed)
```

---

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Verify CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
python3 -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB')"
```

### Out of Memory
- This should NOT happen on RTX 6000 (96GB)
- If it does, reduce `batch_size` in `finetune_qwen_lora.py` from 64 → 32

### Training Interrupted
```bash
# Check for checkpoint files
ls -lah qwen_gb_detector_lora/checkpoint-*/

# Resume from checkpoint (manual):
# Edit finetune_qwen_lora.py and set resume_from_checkpoint parameter
```

---

## 📊 Output Files

After successful training:

```
qwen_gb_detector_lora/
├── adapter_config.json        [LoRA configuration]
├── adapter_model.bin          [LoRA weights (~200 MB)]
├── training_args.bin          [Training configuration]
├── trainer_state.json         [Training state & metrics]
├── checkpoint-1/              [Epoch 1 checkpoint]
├── checkpoint-2/              [Epoch 2 checkpoint]
├── checkpoint-3/              [Epoch 3 checkpoint]
└── runs/tensorboard/          [TensorBoard logs]
```

---

## 🎯 Post-Training Steps

### 1. Validate Model
```bash
python3 manage.py inference
# Test with sample Thai text containing gender bias
```

### 2. Export for Deployment
```bash
# Copy trained model to deployment location
cp -r qwen_gb_detector_lora/ /path/to/deployment/models/
```

### 3. Run Batch Inference (Optional)
```bash
# Process JSONL file with test data
python3 inference_qwen_span.py --mode batch --input test_samples.jsonl
```

---

## 📝 Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen 3.5 2B-Instruct | Optimized for Thai |
| LoRA Rank | 128 | Sufficient for task complexity |
| LoRA Alpha | 256 | Standard 2x multiplier |
| Batch Size | 64 | High due to 96GB VRAM |
| Learning Rate | 2e-4 | Conservative default |
| Epochs | 3 | 3 passes over ~23k samples |
| Optimizer | adamw_8bit | Memory-efficient |
| FP | 16-bit | Not quantized (full LoRA) |

---

## 🔐 Security Notes

- ⚠️ No credentials or secrets stored
- ✅ All data synthetic (from `synthesizer_v3`)
- ✅ No external API calls
- ✅ Fully offline after model download

---

## 📞 Support

For issues:
1. Check START_HERE.txt for quick reference
2. Check SERVICE_README.md for detailed usage
3. Run validation: `python3 validate_system_prompt.py`
4. Check training logs in `runs/` directory

---

**Last Updated**: April 22, 2026  
**Deployed By**: OpenCode Agent  
**Status**: Ready for Production ✅
