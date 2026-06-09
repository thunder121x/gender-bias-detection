# New Training Data Generation Workflow

## Overview

This document describes the new data generation and preparation workflow for training the Qwen LoRA model on gender bias detection.

**New Format**: Instruction-Input-Output → ChatML Conversion

## Data Flow

```
services/finetuning/
├── data/
│   ├── train.jsonl (original, ~80MB)
│   └── instruction_train.jsonl (NEW! 10k samples)
├── scripts/
│   ├── 01_generate_data.py (original, generates train.jsonl)
│   ├── 02_generate_instruction_data.py (NEW! generates instruction format)
│   └── 03_prepare_instruction_data.py (NEW! converts to ChatML)

                    ↓ (02_generate_instruction_data.py)

services/finetuning/data/
├── instruction_train.jsonl (9.5k samples)
├── instruction_val.jsonl (0.5k samples)

                    ↓ (03_prepare_instruction_data.py)

services/finetuning/data/
├── train_chatml.jsonl (ChatML format)
├── val_chatml.jsonl (ChatML format)

                    ↓ (copy to lora_finetuning)

services/lora_finetuning/training_data/
├── train.jsonl (30MB, ready for training)
├── val.jsonl (1.6MB, ready for validation)
```

## Step-by-Step Usage

### 1. Generate Instruction-Input-Output Format Data

```bash
cd services/finetuning
python3 scripts/02_generate_instruction_data.py --num-samples 10000 --output-dir data/ --split
```

**Output**:
- `data/instruction_train.jsonl` (9500 samples)
- `data/instruction_val.jsonl` (500 samples)

**Format** (instruction-input-output):
```json
{
  "instruction": "Identify and tag social bias in the following Thai text...",
  "input": "ผู้หญิงทุกคนก็โง่ สวัสดี หีอย่างอูม",
  "output": "<GB-NORM>ผู้หญิงทุกคนก็โง่</GB-NORM>สวัสดี<GB-SEX>หีอย่างอูม</GB-SEX>"
}
```

### 2. Convert to ChatML Format

```bash
cd services/finetuning

# Convert training data
python3 scripts/03_prepare_instruction_data.py \
  --input data/instruction_train.jsonl \
  --output data/train_chatml.jsonl

# Convert validation data
python3 scripts/03_prepare_instruction_data.py \
  --input data/instruction_val.jsonl \
  --output data/val_chatml.jsonl
```

**Output**:
- `data/train_chatml.jsonl` (ChatML format)
- `data/val_chatml.jsonl` (ChatML format)

**Format** (ChatML):
```json
{
  "text": "<s>[INST] <<SYS>>\nYou are an expert Thai language analyst specializing in detecting and tagging gender bias...\n<</SYS>>\n\nInstruction: Identify and tag social bias...\nInput: ผู้หญิงทุกคนก็โง่ สวัสดี หีอย่างอูม\n[/INST] <GB-NORM>ผู้หญิงทุกคนก็โง่</GB-NORM>สวัสดี<GB-SEX>หีอย่างอูม</GB-SEX></s>",
  "instruction": "Identify and tag social bias...",
  "input": "ผู้หญิงทุกคนก็โง่ สวัสดี หีอย่างอูม",
  "output": "<GB-NORM>ผู้หญิงทุกคนก็โง่</GB-NORM>สวัสดี<GB-SEX>หีอย่างอูม</GB-SEX>"
}
```

### 3. Copy to LoRA Fine-tuning Service

```bash
cp services/finetuning/data/train_chatml.jsonl \
   services/lora_finetuning/training_data/train.jsonl

cp services/finetuning/data/val_chatml.jsonl \
   services/lora_finetuning/training_data/val.jsonl
```

### 4. Run LoRA Fine-tuning

```bash
cd services/lora_finetuning
python3 manage.py
# Select: 3 (train)
```

---

## Data Characteristics

### Instruction Format

**Instruction**: Clear directive to the model
```
Identify and tag social bias in the following Thai text using <GB-NORM> for generalized bias, <GB-ATTACK> for personal attacks, and <GB-SEX> for sexual harassment.
```

**Input**: Original Thai text (may contain 1+ biases or no bias)
```
ผู้หญิงทุกคนก็โง่ สวัสดี หีอย่างอูม
```

**Output**: Same text with bias spans tagged
```
<GB-NORM>ผู้หญิงทุกคนก็โง่</GB-NORM>สวัสดี<GB-SEX>หีอย่างอูม</GB-SEX>
```

### Data Statistics

- **Total samples**: 10,000
- **Train/Val split**: 95/5 (9,500 / 500)
- **Bias types**: 
  - GB-NORM (Generalized bias/stereotypes)
  - GB-ATTACK (Personal attacks)
  - GB-SEX (Sexual harassment)
- **Also includes**: Neutral sentences with no bias
- **Sentences per sample**: 2-5 sentences (mixed bias/neutral)

### Tag Meanings

| Tag | Meaning | Example |
|-----|---------|---------|
| `<GB-NORM>` | Generalized gender bias, stereotypes | "ผู้หญิงทุกคนก็โง่" (all women are dumb) |
| `<GB-ATTACK>` | Personal attacks based on gender | "ผู้ชายคนนั้นเห็นแก่ตัว" (that man is selfish) |
| `<GB-SEX>` | Sexual harassment, objectification | "หีอย่างอูม" (crude sexual reference) |

---

## Notes

1. **Mixed Content**: Each text sample may contain:
   - 0 biased spans (only neutral)
   - 1 biased span (one gender bias)
   - 2+ biased spans (multiple different biases)

2. **Real-world Complexity**: Texts are generated from real comments, preserving:
   - Mixed-language patterns (Thai with English/numbers)
   - Multiple topics in one text (natural conversation flow)
   - Neutral fillers between biased spans

3. **Format Flexibility**: The ChatML format allows:
   - System prompt customization
   - Different task descriptions
   - Clear instruction-output separation

---

## Troubleshooting

### Script not found
```bash
# Ensure you're in the right directory
cd services/finetuning
ls scripts/
```

### No CUDA/GPU
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Memory issues during generation
```bash
# Reduce number of samples
python3 scripts/02_generate_instruction_data.py --num-samples 1000
```

---

## Regenerating Data

To regenerate the data with different parameters:

```bash
# Generate 20,000 samples instead of 10,000
cd services/finetuning
python3 scripts/02_generate_instruction_data.py --num-samples 20000 --output-dir data/ --split

# Convert to ChatML
python3 scripts/03_prepare_instruction_data.py --input data/instruction_train.jsonl --output data/train_chatml.jsonl
python3 scripts/03_prepare_instruction_data.py --input data/instruction_val.jsonl --output data/val_chatml.jsonl

# Copy to lora_finetuning
cp data/train_chatml.jsonl ../lora_finetuning/training_data/train.jsonl
cp data/val_chatml.jsonl ../lora_finetuning/training_data/val.jsonl
```

---

**Last Updated**: April 22, 2026
