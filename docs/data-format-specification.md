# Fine-tuning Data Format Specification (NEW VERSION)

## ⚡ Important: Data Format Update

The training data has been **converted to a new JSON schema format** optimized for fine-tuning with chat-style LLMs.

**Old format** (XML tags): ❌ NO LONGER USED  
**New format** (JSON schema): ✅ NOW IN USE

---

## 📋 Data Format Overview

### Training Data Structure

Each line in `train.jsonl` and `val.jsonl` contains:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt describing the task..."
    },
    {
      "role": "user",
      "content": "Input text to analyze for gender bias..."
    },
    {
      "role": "assistant",
      "content": "{\"spans\": [{\"label\": \"GB-TYPE\", \"text\": \"...\"}, ...]}"
    }
  ]
}
```

### Assistant Response Format (JSON Schema)

The assistant always outputs structured JSON:

```json
{
  "spans": [
    {
      "label": "GB-ATTACK",
      "text": "Text from the input containing gender bias attack"
    },
    {
      "label": "GB-NORMATIVE",
      "text": "Text containing gender stereotype or role enforcement"
    },
    {
      "label": "GB-SEX",
      "text": "Text with sexualized or sexual harassment content"
    }
  ]
}
```

**Or if no bias found:**
```json
{
  "spans": []
}
```

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Examples** | 10,000 |
| **Training Examples** | 9,500 |
| **Validation Examples** | 500 |
| **Examples with Bias** | 6,339 (63.4%) |
| **Examples without Bias** | 3,161 (31.6%) |
| **Total Spans** | 9,469 |
| **Average Spans per Example** | 1.00 |
| **Average Span Length** | 52.2 characters |

### Label Distribution

| Label | Count | Percentage |
|-------|-------|-----------|
| **GB-ATTACK** | 3,114 | 32.9% |
| **GB-NORMATIVE** | 3,119 | 32.9% |
| **GB-SEX** | 3,236 | 34.2% |

### Span Count Distribution

| Spans | Examples | Percentage |
|-------|----------|-----------|
| **0 spans** (neutral) | 3,161 | 33.3% |
| **1 span** | 3,209 | 33.8% |
| **2 spans** | 3,130 | 32.9% |

---

## 🏷️ Gender Bias Labels

### GB-ATTACK
**Direct attacks, derogatory language, dehumanization**

Examples:
- "ไอ้ตุ๊ดพวกนี้ไม่นับเป็นคน" (These fairies don't count as human)
- "ผู้หญิงมันโง่หมดทุกคน" (All women are stupid)
- "กะเทยคือของเสียสังคม" (Gays are society's waste)

Characteristics:
- Insulting, derogatory, or dehumanizing language
- Genetic/inherent slurs about gender groups
- Direct attacks on group intelligence, morality, or worth

### GB-NORMATIVE
**Stereotypes, gender roles, benevolent bias**

Examples:
- "ผู้ชายต้องเป็นผู้นำเสมอ" (Men must always be leaders)
- "ทอมที่แท้จริงต้องแมนกว่านี้" (A real tom must be more masculine)
- "ผู้หญิงโดยธรรมชาติอ่อนโยน" (Women are naturally gentle)

Characteristics:
- Stereotyping specific groups
- Enforcing gender roles or expectations
- Policing gender identity or expression
- Benevolent bias (positive but restrictive stereotypes)

### GB-SEX
**Sexualization, body-shaming, sexual harassment**

Examples:
- "หีดำแบบนี้มันไม่ใช่ผู้หญิงดี ๆ" (A dark vagina isn't a good woman)
- "ควยเล็กแล้วจะเป็นผู้ชายได้ไง" (How can you be a man with a small penis)
- "กะเทยนมปลอมเหมือนบอลลูน" (Transgender women's fake boobs are like balloons)

Characteristics:
- Sexual objectification or harassment
- Body-shaming tied to gender/sex
- Sexual performance shame attacking gender groups
- Sexualized insults about gender groups

---

## 💾 File Locations

```
services/lora_finetuning/training_data/
├── train.jsonl                          # 9,500 examples
├── val.jsonl                            # 500 examples
├── train_old_tagbased.jsonl            # Backup (old format)
└── val_old_tagbased.jsonl              # Backup (old format)
```

**File Sizes:**
- `train.jsonl`: 89.4 MB
- `val.jsonl`: 4.7 MB
- **Total**: 94.1 MB (15% smaller than old format)

---

## 🔄 Format Comparison

### Old Format (DEPRECATED)
```
<GB-NORMATIVE>ผู้ชายต้องเป็นผู้นำเสมอ</GB-NORMATIVE> <GB-ATTACK>ไอ้ตุ๊ดนี่ไม่นับเป็นคน</GB-ATTACK>
```

### New Format (CURRENT)
```json
{
  "messages": [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "Input: ผู้ชายต้องเป็นผู้นำเสมอ ไอ้ตุ๊ดนี่ไม่นับเป็นคน"},
    {"role": "assistant", "content": "{\"spans\": [{\"label\": \"GB-NORMATIVE\", \"text\": \"ผู้ชายต้องเป็นผู้นำเสมอ\"}, {\"label\": \"GB-ATTACK\", \"text\": \"ไอ้ตุ๊ดนี่ไม่นับเป็นคน\"}]}"}
  ]
}
```

**Advantages of New Format:**
✅ Chat-style format (compatible with more models)  
✅ Structured JSON output (easier to parse)  
✅ Cleaner message separation (system/user/assistant)  
✅ Better for instruction-following fine-tuning  
✅ Reduced token overhead (15% smaller)  
✅ Easier to extend with additional fields  

---

## 📖 Using the Data in Training Scripts

The fine-tuning scripts automatically handle the new format:

```python
# Load data (scripts handle this automatically)
dataset = GenderBiasDataset(
    file_path="services/lora_finetuning/training_data/train.jsonl",
    tokenizer=tokenizer,
    max_seq_length=2048
)

# Each example returns:
{
    "input_ids": tensor,           # Tokenized conversation
    "attention_mask": tensor,      # Attention mask
    "labels": tensor,              # Labels for loss calculation
}
```

---

## ✅ Data Quality Assurance

### Validation Performed
✅ All 9,500 training examples validated  
✅ All 500 validation examples validated  
✅ No JSON parsing errors  
✅ All messages have correct structure (system/user/assistant)  
✅ All spans have valid labels (GB-ATTACK, GB-NORMATIVE, GB-SEX)  
✅ All span text exists and is non-empty  

### Data Integrity
- ✅ No examples were lost during conversion
- ✅ All spans preserved from original format
- ✅ Character encoding maintained (UTF-8 Thai text)
- ✅ Whitespace and formatting preserved

---

## 🚀 Starting Training with New Format

```bash
# The training scripts automatically use the new format
python3 train_simple.py --preset balanced

# Data is loaded from:
# - training: services/lora_finetuning/training_data/train.jsonl
# - validation: services/lora_finetuning/training_data/val.jsonl
```

---

## 📝 Example Training Session

When you run training, the script will:

1. ✓ Load 9,500 training examples in new JSON format
2. ✓ Load 500 validation examples in new JSON format
3. ✓ Parse each example's messages (system/user/assistant)
4. ✓ Tokenize the conversation for training
5. ✓ Format assistant response as structured JSON
6. ✓ Train LoRA adapter on gender bias detection task

**Expected behavior:**
- Each epoch processes 9,500 examples
- Validation runs on 500 examples every 100 steps
- Model learns to extract gender bias spans in JSON format
- Output: Fine-tuned model that generates JSON responses

---

## 🔗 Related Files

- **`DATASET_CONVERSION_NOTES.md`** - Conversion process details
- **`convert_dataset.py`** - Conversion script (already run)
- **`inspect_data.py`** - Data inspection utility
- **`train_simple.py`** - Training script (uses new format automatically)
- **`H100_TRAINING_GUIDE.md`** - Complete training guide

---

## ⚠️ Important Notes

1. **Old backup files are preserved**: 
   - `train_old_tagbased.jsonl` - Original tag-based format
   - `val_old_tagbased.jsonl` - Original tag-based format
   - Keep these for reference only

2. **Use new files for training**:
   - Always use `train.jsonl` (not old version)
   - Always use `val.jsonl` (not old version)

3. **Training scripts are already configured**:
   - No manual data format specification needed
   - Scripts automatically handle JSON parsing
   - Just run: `python3 train_simple.py --preset balanced`

---

**Data Conversion Date**: 2026-05-03  
**Format Version**: 2.0 (JSON Schema)  
**Status**: ✅ Ready for Production Training
