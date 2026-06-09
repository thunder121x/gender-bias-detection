# 🎯 Gender Bias Detection Fine-tuning Instruction (v2.0)

## Complete Guide for Training Thai Social Bias Detector

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Task Definition](#task-definition)
3. [Label Structure](#label-structure)
4. [System Prompt](#system-prompt)
5. [Decision Framework](#decision-framework)
6. [Classification Guidelines](#classification-guidelines)
7. [Example Dataset Format](#example-dataset-format)
8. [Training Protocol](#training-protocol)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Common Edge Cases](#common-edge-cases)

---

## 📌 Overview

This instruction set is designed for fine-tuning Thai language models to detect gender bias (GB) and gender-based harassment in Thai social media content. The system classifies text into:

- **GB (Gender Bias)**: Text containing gender-based discriminatory content
- **NON-GB**: Text that does not contain gender bias
- **3 Subtypes**: GB-ATTACK, GB-NORMATIVE, GB-SEX

**Key Characteristics:**
- Binary + Multi-class hierarchical classification
- Focus on Thai context and cultural nuances
- Support for LGBTQ+ (SOGI) bias detection
- Includes Meta/Counter-speech handling

---

## 🎬 Task Definition

**Task Name:** Thai Gender Bias Detection with Trigger & Rationale Extraction

**Input:** Thai text snippet/sentence/paragraph
**Output:** 
```json
{
  "text": "ข้อความต้นฉบับ",
  "label": "GB or NON-GB",
  "subtype": "GB-ATTACK | GB-NORMATIVE | GB-SEX | null",
  "trigger": ["span1", "span2"],
  "rationale": "explanation span",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation"
}
```

---

## 🏷️ Label Structure

### Level 1: Binary Classification

| Label | Code | Definition |
|-------|------|-----------|
| **Gender Bias** | GB | ข้อความที่มีอคติทางเพศตามเกณฑ์ A+B และไม่เข้า C/D |
| **Non Gender Bias** | NON-GB | ข้อความที่ไม่เข้าเกณฑ์ GB |

### Level 2: Bias Subtype (Only for GB)

| Subtype | Thai Name | Definition |
|---------|-----------|-----------|
| **GB-ATTACK** | Attack on Gender | โจมตี ดูหมิ่น ลดคุณค่าเพศอย่างตรงไปตรงมา |
| **GB-NORMATIVE** | Gender Norm Enforcement | เหมารวม กำหนดบทบาท หรือ policing เพศ |
| **GB-SEX** | Sexualized Gender Bias | ลดคุณค่าผ่าน sexualized insult หรือ body-based attack |

---

## 💬 System Prompt

### Main System Role

```
You are a Thai Social Bias Detector. Your role is to identify and classify text containing gender bias (Gender Bias Detection - GB) and gender-related harassment in Thai language.

Use the following tags when explaining your classification:
- <GB-NORM>: For Gender Normative/Gender Role Enforcement bias
- <GB-SEX>: For Sexualized or Body-Based Gender Bias
- <GB-ATTACK>: For Direct Attacks on Gender Groups

Classify each input into one of two categories:
1. GB (Gender Bias) - with subtype: GB-ATTACK, GB-NORMATIVE, or GB-SEX
2. NON-GB (Non Gender Bias) - including meta-commentary, counter-speech, or no gender bias

Always explain your reasoning based on the decision framework criteria (A, B, C, D).
```

### User-Facing Instruction (Thai)

```
จงระบุและใส่แท็กข้อความที่มีอคติทางเพศ (Social Bias) โดยใช้:
- <GB-NORM> สำหรับอคติเกี่ยวกับบทบาทหรือค่านิยมทางเพศ
- <GB-SEX> สำหรับการคุกคามทางเพศ การดูหมิ่นร่างกาย หรือ sexualized insult
- <GB-ATTACK> สำหรับการโจมตีโดยตรงต่อกลุ่มเพศ

ระบุเสมอ:
1. ประเภทอคติ (GB-ATTACK, GB-NORMATIVE, GB-SEX)
2. ส่วนข้อความสำคัญ (Trigger) ที่บ่งบอกอคติ
3. เหตุผลประกอบ (Rationale) ในการจำแนกประเภท
4. ข้อมูลเพิ่มเติม เช่น identity targeting, severity, context
```

---

## 🔍 Decision Framework

### Criterion (A): Gendered Target - มีการอ้างอิงเพศ/SOGI

Text must contain explicit or implicit reference to gender/sexual orientation/gender identity

#### A1: Explicit Reference (อ้างอิงโดยตรง)
- ผู้หญิง, ผญ, ผู้ชาย, ผช, ชายแท้, หญิงแท้, เมีย, ผัว
- เกย์, เลสเบี้ยน, ไบ, ทอม, ดี้, ตุ๊ด, กะเทย, สาวสอง

#### A2: Implicit Reference (อ้างอิงโดยอ้อม)
- Through sexual organs or sexual behaviors: "พวกใช้หัวควย…" → male
- Through cultural roles with gendered connotation: "เฟมทวิต" → female activist group
- Through gender-coded slang: "ถั่วดำ" → gay sex behavior
- Through gendered characteristics: "นมปลอม" → transgender women

#### A3: Inherently Gendered Slur
Use "slur test": Does the slur inherently demean a gender group even if used against one person?

**Examples:**
- Woman-Degrading Slurs: "กะหรี่", "ร่าน", "โสเภณี", "สุนัขตัวเมีย"
- Anti-LGBT Slurs: "อีตุ๊ด", "ลักเพศ", "ผิดเพศ", "ถังขี้"
- Sexualized Slurs: "จู๋เล็ก", "หอยเน่า", "ตูดหลวม", "นมปลอม"

**✓ Passes Criterion A** if ANY of A1, A2, or A3 applies

---

### Criterion (B): Negative Evaluation - การประเมินค่าลบต่อกลุ่มเพศ

Text must contain negative judgment, derogation, or stereotyping toward gender group

#### B1: Insulting/Derogatory/Dehumanizing
```
ผู้หญิงมันโง่หมดทุกคน
→ Uses derogatory language toward women as a group
→ "โง่" = insulting + "ทุกคน" = generalization
```

#### B2: Negative Generalization
```
ผู้ชายแม่งเหี้ยเหมือนกันหมด
→ Stereotyping all men with negative trait
→ Applies to "entire gender" (Key: ทุกคน/เหมือนกันหมด/พวกนี้)
```

#### B3: Justification of Harm
```
ผู้หญิงแบบนี้สมควรโดนทำร้าย
→ Says gender group "deserves" harm
→ Justifies violence or sexual abuse
```

#### B4: Value Tied to Sexual Traits
```
ควยเล็กแล้วจะเป็นผู้ชายได้ไง
→ Links manhood/male worth to penis size
→ Reduces gender value to sexual traits
```

#### B5: Positive/Benevolent Stereotype
```
ผู้หญิงโดยธรรมชาติอ่อนโยน
→ Assigns fixed traits to entire gender
→ Restricts gender expression to narrow roles
```

**✓ Passes Criterion B** if ANY of B1-B5 applies AND is directed at gender group

---

### Criterion (C): Meta/Counter-Speech - ไม่เป็น GB

If text is EXPLICITLY analyzing, criticizing, or opposing gender bias, mark as **NON-GB**

#### C1: Meta Commentary
```
ยังมีคนพูดว่าผู้หญิงโง่อยู่เลย น่าสมเพชมาก
→ Quoting someone else's bias
→ Using words like "มีคนพูดว่า", "บางคนคิด", "เคยได้ยิน"
```

#### C2: Social Critique
```
สังคมไทยชอบโทษผู้หญิงเวลาโดนคุกคามทางเพศ
→ Criticizing society's structure
→ NOT endorsing the bias
```

#### C3: Counter-Speech
```
ใครบอกว่าผู้หญิงขับรถไม่เก่ง นั่นคืออคติล้วน ๆ
→ Explicitly rejecting bias
→ Using words like "นั่นคืออคติ", "ผิด", "ไม่ถูก"
```

**Key Test:** 
- If speaker is ENDORSING bias → **GB**
- If speaker is DESCRIBING/CRITICIZING bias → **NON-GB (C)**

---

### Criterion (D): Gendered Insult - Individual Target

If text insults ONE person using gender reference but doesn't generalize to whole gender group

#### D Requirements (ALL must be true):
1. Targets SPECIFIC person (e.g., "คนนั้น", "พี่คนนี้", "เขา")
2. Uses negative word but NOT a gendered slur
3. No generalization to entire gender group
4. Gender is only used to identify person, not as reason for insult

#### D Examples:
```
"คนนั้นอะ ผู้ชายโง่ๆ"
→ Insult is about that specific man, not ALL men
→ "โง่ๆ" is not a gender slur
→ Gender is referential, not the basis of insult

"ผู้หญิงคนนั้น ขี้วีนมาก"
→ Complaining about individual woman
→ "ขี้วีน" is personality trait, not gendered attack
→ No stereotyping
```

**✓ Passes Criterion D** if INDIVIDUAL focus + NO SLUR + NO GENERALIZATION

**❌ FAILS Criterion D** if:
- "กะหรี่คนนั้น" → Using gendered slur
- "ผู้หญิงพวกนี้ขี้วีน" → Generalizing to group

---

## 📊 Classification Guidelines

### Decision Tree

```
Is text gender/SOGI related?
├─ NO → NEUTRAL (no GB/NON-GB label)
│
└─ YES
   ├─ Does it pass Criterion A (gendered target)?
   │  ├─ NO → NEUTRAL
   │  │
   │  └─ YES → Check Criterion B (negative evaluation)
   │     ├─ NO → NEUTRAL
   │     │
   │     └─ YES → Check Criterion C (meta/counter)
   │        ├─ YES (C1/C2/C3) → NON-GB (C)
   │        │
   │        └─ NO → Check Criterion D (individual gendered insult)
   │           ├─ YES (all D requirements) → NON-GB (D)
   │           │
   │           └─ NO → **GB** + Assign Subtype (A→SEX, N→NORM, T→ATTACK)
```

### Subtype Selection for GB Classification

**GB-ATTACK** when:
- Direct insult with derogatory/slur language
- Dehumanizing language
- Anti-LGBTQ+ slurs
- Calls for harm toward gender group

Examples:
```
ผู้หญิงมันโง่หมดทุกคน
ไอ้ตุ๊ดพวกนี้ไม่นับเป็นคน
ผู้ชายแม่งเหี้ยหมด
```

**GB-NORMATIVE** when:
- Stereotyping (positive or negative)
- Gender role enforcement
- Identity policing
- Normalcy judgments

Examples:
```
ผู้ชายต้องเป็นผู้นำเสมอ
ทอมที่แท้จริงต้องแมนกว่านี้
ผู้หญิงโดยธรรมชาติหวาน
กะเทยเป็นคนขี้เม้าท์ทุกคน
```

**GB-SEX** when:
- Sexualized insults
- Body-based insults tied to gender
- Sexual shame or coercion
- Reduces worth to sexual traits

Examples:
```
ควยเล็กแล้วจะเป็นผู้ชายได้ไง
หีดำแบบนี้มันไม่ใช่ผู้หญิงดี ๆ
นมปลอมเหมือนบอลลูน
ตูดหลวมไม่มีค่า
```

---

## 📝 Example Dataset Format

### Format 1: Simple CSV

```csv
text,label,subtype,trigger,rationale
"ผู้หญิงมันโง่หมดทุกคน",GB,GB-ATTACK,"ผู้หญิง","สรุปลบกับเพศหญิงทั้งหมด"
"สังคมไทยกดทับผู้หญิงเยอะมาก",NON-GB,,,"วิพากษ์โครงสร้างสังคม ไม่ใช่การเหยียดของผู้พูด"
"ผู้ชายต้องเป็นผู้นำเสมอ",GB,GB-NORMATIVE,"ผู้ชาย,ต้อง,เสมอ","กำหนดบทบาทเพศชาย"
```

### Format 2: Detailed JSON

```json
[
  {
    "id": "001",
    "text": "ผู้หญิงมันโง่หมดทุกคน",
    "label": "GB",
    "subtype": "GB-ATTACK",
    "criteria_met": ["A", "B"],
    "criteria_failed": ["C", "D"],
    "trigger": ["ผู้หญิง", "โง่", "ทุกคน"],
    "rationale": "สรุปลบเกี่ยวกับเพศหญิงทั้งหมดด้วยคำดูหมิ่น",
    "reasoning": {
      "A": "explicitly references women",
      "B": "negative generalization ('โง่' applied to all)",
      "C": "not meta/counter-speech",
      "D": "not individual insult - generalizes to all women"
    },
    "severity": "high",
    "target_group": "women",
    "language_type": "slur_derogatory"
  },
  {
    "id": "002",
    "text": "คนนั้นอะ ผู้ชายโง่ๆ",
    "label": "NON-GB",
    "subtype": null,
    "criteria_met": ["A", "D"],
    "criteria_failed": ["B", "C"],
    "trigger": ["คนนั้น"],
    "rationale": "ด่าบุคคลเดียวโดยใช้เพศประกอบ ไม่ได้เหมารวมเพศชาย",
    "reasoning": {
      "A": "mentions gender (ผู้ชาย)",
      "B_failed": "insult targets individual, not male group",
      "D": "meets all gendered insult criteria"
    },
    "severity": "low",
    "reason_nongb": "gendered_insult"
  },
  {
    "id": "003",
    "text": "ยังมีคนพูดว่าผู้หญิงควรอยู่บ้าน น่าสมเพชจริง ๆ",
    "label": "NON-GB",
    "subtype": null,
    "criteria_met": ["A", "C"],
    "criteria_failed": ["B"],
    "trigger": ["ยังมีคนพูดว่า", "น่าสมเพช"],
    "rationale": "กล่าวถึงการเหยียดของผู้อื่น พร้อมทำการวิพากษ์ (meta-commentary)",
    "reasoning": {
      "A": "references women and their roles",
      "C": "speaker is criticizing others' bias, not endorsing it",
      "meta_indicator": "ยังมีคนพูด + น่าสมเพช shows criticism"
    },
    "severity": "low",
    "reason_nongb": "meta_commentary"
  },
  {
    "id": "004",
    "text": "ทอมที่แท้จริงต้องแมนกว่านี้",
    "label": "GB",
    "subtype": "GB-NORMATIVE",
    "criteria_met": ["A", "B"],
    "criteria_failed": ["C", "D"],
    "trigger": ["ทอม", "ต้อง", "แมน"],
    "rationale": "policing เพศสภาพและเหมารวมลักษณะของทอม",
    "reasoning": {
      "A": "implicitly references tom (gender identity)",
      "B": "enforces gender norm/identity - 'real toms must be more masculine'",
      "type": "identity policing"
    },
    "severity": "medium",
    "target_group": "tom identity",
    "language_type": "norm_enforcement"
  }
]
```

### Format 3: Conversation Format (for Instruction Tuning)

```
USER: จำแนกข้อความต่อไปนี้: "ผู้หญิงมันโง่หมดทุกคน"

ASSISTANT: ข้อความนี้มีอคติทางเพศประเภท <GB-ATTACK>

**การวิเคราะห์:**

1. **เกณฑ์ A (Gendered Target):** ✓ ผ่าน
   - มีการอ้างอิงเพศชัดเจน: "ผู้หญิง"

2. **เกณฑ์ B (Negative Evaluation):** ✓ ผ่าน
   - ใช้คำดูหมิ่น: "โง่"
   - เหมารวมเป็นกลุ่ม: "ทุกคน"
   - ลดคุณค่าเพศหญิง

3. **เกณฑ์ C (Meta/Counter):** ✗ ไม่ผ่าน
   - ไม่มีการกล่าวถึงความเห็นของผู้อื่น
   - ไม่มีการวิพากษ์โครงสร้างสังคม
   - ผู้พูดเป็นผู้เหยียดตรงๆ

4. **เกณฑ์ D (Individual Insult):** ✗ ไม่ผ่าน
   - คำว่า "ทุกคน" แสดงว่าเป็นการสรุปทั้งกลุ่ม ไม่ใช่บุคคลเดียว

**ผลการจำแนก:** GB-ATTACK
**Trigger:** ผู้หญิง, โง่, ทุกคน
**Severity:** High
```

---

## 🚀 Training Protocol

### Data Preparation

```python
# Step 1: Data Preprocessing
1. Clean Thai text (normalize whitespace, fix encoding)
2. Remove PII (personal identifiers)
3. Segment into sentences if needed
4. Create balanced dataset:
   - 40% GB examples (10% each subtype + 10% hard cases)
   - 30% NON-GB examples (10% meta, 10% gendered-insult, 10% neutral)
   - 30% Neutral examples

# Step 2: Data Annotation
- Each text annotated by 3 raters (minimum)
- Resolve disagreements through discussion
- Calculate inter-rater reliability (Cohen's Kappa ≥ 0.75)
- Store: text, label, subtype, trigger, rationale, confidence
```

### Fine-tuning Configuration

```python
# Recommended Parameters for LLM Fine-tuning

MODEL_CONFIG = {
    # Model selection
    "base_model": "mistral-7b-instruct",  # or other Thai-capable models
    "model_type": "instruction_tuned",
    
    # LoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    
    # Training parameters
    "learning_rate": 1e-4,
    "batch_size": 8,
    "num_epochs": 3,
    "max_seq_length": 512,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,
    
    # Loss function
    "loss_type": "multi_task",  # For label + subtype + spans
    "class_weights": {
        "GB": 1.5,
        "NON-GB": 1.0,
        "GB-ATTACK": 1.2,
        "GB-NORMATIVE": 1.2,
        "GB-SEX": 1.2
    }
}

TRAINING_PROMPT_TEMPLATE = """<|im_start|>system
You are a Thai Gender Bias Detector. Classify the given Thai text and identify bias indicators.
- Use criterion A (gendered target), B (negative evaluation), C (meta-speech), D (individual insult)
- Label: GB or NON-GB
- If GB, specify subtype: GB-ATTACK, GB-NORMATIVE, or GB-SEX
- Extract trigger phrases and provide rationale
<|im_end|>

<|im_start|>user
Text: {text}
<|im_end|>

<|im_start|>assistant
Label: {label}
Subtype: {subtype}
Trigger: {trigger}
Rationale: {rationale}
Reasoning: {reasoning}
<|im_end|>"""
```

### Evaluation Protocol

```python
# Evaluation Metrics

EVALUATION = {
    "metrics": {
        # Classification metrics
        "accuracy": "exact label match",
        "f1_macro": "macro F1 across all classes",
        "precision_per_class": "class-wise precision",
        "recall_per_class": "class-wise recall",
        
        # Subtype accuracy (only for GB)
        "subtype_accuracy": "accuracy of subtype classification given GB",
        
        # Span extraction metrics
        "trigger_precision": "% of extracted triggers that are correct",
        "trigger_recall": "% of gold triggers that are extracted",
        
        # Rationale quality
        "rationale_bleu": "BLEU score for rationale generation",
        
        # Edge case handling
        "edge_case_accuracy": {
            "meta_commentary": "accuracy on meta/counter-speech",
            "gendered_insult": "accuracy on individual gendered insults",
            "implicit_reference": "accuracy on implicit gender reference",
            "slur_detection": "accuracy on gendered slur detection"
        }
    },
    
    "test_split_strategy": "stratified by label and subtype",
    "evaluation_set_size": 500,  # minimum
}
```

---

## 📈 Evaluation Metrics

### Performance Benchmarks

```
Target Performance:
├─ Overall Accuracy: ≥ 85%
├─ GB Precision: ≥ 88%
├─ GB Recall: ≥ 82%
├─ Subtype F1 (per class): ≥ 80%
├─ NON-GB Accuracy: ≥ 85%
└─ Edge Case Handling:
   ├─ Meta-commentary detection: ≥ 90%
   ├─ Gendered-insult detection: ≥ 85%
   └─ Slur recognition: ≥ 92%
```

### Confusion Matrix Analysis

Focus on:
- GB ↔ NON-GB misclassification
- Subtype confusion (especially GB-NORMATIVE vs GB-ATTACK)
- False positives on neutral gendered content
- False negatives on implicit references

---

## ⚠️ Common Edge Cases

### Case 1: Ambiguous Gender Reference

```
"ทำไมถึงต้องแต่งตัวแบบนั้นล่ะ"
→ NEUTRAL (no gendered target)
→ "นั้น" is referential, not gendered
→ No explicit/implicit gender group reference

BUT:

"ทำไมผู้หญิงถึงต้องแต่งตัวแบบนั้น"
→ GB-NORMATIVE (gender role enforcement)
→ "ผู้หญิง" = gendered target
→ "ต้อง" = norm enforcement
```

### Case 2: Positive Stereotype

```
"ผู้หญิงโดยธรรมชาติหวาน"
→ GB-NORMATIVE
→ Criterion B5: Benevolent stereotype
→ Still GB because it restricts gender expression
→ Even though it seems complimentary
```

### Case 3: Slur Usage in Quote

```
"ยังมีคนบอกว่า 'ไอ้ตุ๊ดพวกนี้ไม่นับเป็นคน' น่าสมเพช"
→ NON-GB
→ Criterion C: Meta-commentary
→ Speaker is quoting and criticizing, not endorsing
→ Look for indicators: "บอกว่า" + negative sentiment marker ("น่าสมเพช")
```

### Case 4: Sexual Content Without Gender Bias

```
"เมื่อคืนมีอะไรกับแฟน"
→ NON-GB / NEUTRAL
→ Sexual content ≠ Gender bias
→ Only mark as GB-SEX if uses gendered insult or sexualized attack
→ Key: Is it consensual/informational or attacking?
```

### Case 5: Identity Policing

```
"นี่ไม่ใช่ผู้หญิงจริง เป็นแค่คนที่เลียนแบบ"
→ GB-NORMATIVE
→ Criterion B: Identity policing
→ Dictating what "real woman" means
→ Applies to trans women, femboys, etc.
```

### Case 6: Implicit Group Reference

```
"พวกสาวเฟม always angry"
→ GB (likely GB-NORMATIVE)
→ "สาวเฟม" = implicitly female activist group
→ "always angry" = negative generalization
→ Implicit cultural reference still triggers Criterion A
```

### Case 7: Severity Gradations

```
"ผู้หญิงไม่ควรออกจากบ้าน"
→ HIGH severity GB-NORMATIVE
→ Strong norm enforcement

"ผู้หญิงเหมือนจะชอบการช็อปปิ้ง"
→ MEDIUM severity GB-NORMATIVE
→ Benevolent stereotype but still restrictive

"ผู้หญิงส่วนใหญ่ชอบการช็อปปิ้ง"
→ LOW severity GB-NORMATIVE?
→ Empirical claim with qualifier "ส่วนใหญ่"
→ Still problematic but less severe
→ **Decision:** Still mark as GB-NORMATIVE (stereotyping exists)
```

---

## 🎓 Instruction Tuning Format

### Few-shot Examples for Model Training

```python
TRAINING_EXAMPLES = [
    # Example 1: Clear GB-ATTACK
    {
        "instruction": "จำแนกข้อความต่อไปนี้และอธิบายเหตุผล",
        "input": "ผู้หญิงมันโง่หมดทุกคน",
        "output": """Label: GB
Subtype: GB-ATTACK
Trigger: ['ผู้หญิง', 'โง่', 'ทุกคน']
Rationale: สรุปลบเกี่ยวกับเพศหญิงทั้งหมด
Reasoning: 
- Criterion A ✓: explicitly references women
- Criterion B ✓: negative generalization (โง่ = insulting, ทุกคน = all women)
- Criterion C ✗: not meta/counter-speech
- Criterion D ✗: not individual insult
Classification: GB-ATTACK (direct derogatory attack on gender group)"""
    },
    
    # Example 2: NON-GB with Meta-commentary
    {
        "instruction": "จำแนกข้อความต่อไปนี้และอธิบายเหตุผล",
        "input": "ยังมีคนพูดว่าผู้หญิงควรอยู่บ้าน น่าสมเพชจริง ๆ",
        "output": """Label: NON-GB
Subtype: null
Trigger: ['ยังมีคนพูดว่า', 'น่าสมเพช']
Rationale: กล่าวถึงการเหยียดของผู้อื่นและทำการวิพากษ์
Reasoning:
- Criterion A ✓: references women and gender roles
- Criterion B ✓: contains negative judgment
- Criterion C ✓: meta-commentary - speaker is criticizing others' bias
- Decision Rule: When C is true, label as NON-GB regardless of A/B
Classification: NON-GB (meta-commentary)"""
    },
    
    # Example 3: NON-GB Gendered-Insult
    {
        "instruction": "จำแนกข้อความต่อไปนี้และอธิบายเหตุผล",
        "input": "คนนั้นอะ ผู้ชายโง่ๆ",
        "output": """Label: NON-GB
Subtype: null
Trigger: ['คนนั้น']
Rationale: ด่าบุคคลเดียว ไม่ได้เหมารวมเพศชาย
Reasoning:
- Criterion A ✓: references man (ผู้ชาย)
- Criterion B ✗: insult targets individual, not all men
- Criterion D ✓: all requirements met
  1. Specific person (คนนั้น)
  2. Negative word that's not gendered slur (โง่ = personality, not gendered slur)
  3. No generalization to all men
  4. Gender is referential only
Classification: NON-GB (gendered-insult)"""
    },
    
    # Example 4: GB-NORMATIVE with Identity Policing
    {
        "instruction": "จำแนกข้อความต่อไปนี้และอธิบายเหตุผล",
        "input": "ทอมที่แท้จริงต้องแมนกว่านี้",
        "output": """Label: GB
Subtype: GB-NORMATIVE
Trigger: ['ทอม', 'ต้อง', 'แมน']
Rationale: policing เพศสภาพและการแสดงออก
Reasoning:
- Criterion A ✓: implicitly references tom (gender identity category)
- Criterion B ✓: identity policing - prescribing how "real toms" should be
- Criterion C ✗: not meta/counter-speech
- Criterion D ✗: generalizes to gender identity group
Type: Identity policing (enforcing gender norms on LGBTQ+ identities)
Classification: GB-NORMATIVE"""
    },
    
    # Example 5: GB-SEX
    {
        "instruction": "จำแนกข้อความต่อไปนี้และอธิบายเหตุผล",
        "input": "ควยเล็กแล้วจะเป็นผู้ชายได้ไง",
        "output": """Label: GB
Subtype: GB-SEX
Trigger: ['ควยเล็ก', 'ผู้ชาย']
Rationale: ลดคุณค่าผู้ชายผ่านอวัยวะเพศ
Reasoning:
- Criterion A ✓: references manhood (ผู้ชาย)
- Criterion B ✓: ties male worth to penis size (sexualized insult)
- Criterion C ✗: not meta/counter-speech
- Criterion D ✗: generalizes to men
Type: Sexualized attack linking gender value to sexual anatomy
Classification: GB-SEX"""
    }
]
```

---

## 📋 Checklist for Implementation

- [ ] System prompt defined in Thai and English
- [ ] All 4 criteria (A/B/C/D) understood and exemplified
- [ ] All 3 subtypes (ATTACK/NORMATIVE/SEX) clearly distinguished
- [ ] Dataset balanced across labels and subtypes
- [ ] Inter-rater reliability calculated (Cohen's Kappa ≥ 0.75)
- [ ] Edge cases documented and handled
- [ ] Few-shot examples prepared for instruction tuning
- [ ] Training/validation/test splits created
- [ ] Performance benchmarks set and agreed upon
- [ ] Error analysis framework prepared
- [ ] Thai language preprocessing implemented
- [ ] Implicit reference detection guidelines finalized
- [ ] Slur dictionary compiled and maintained
- [ ] Model evaluation on Thai test set completed
- [ ] Human evaluation on sample predictions done
- [ ] Deployment readiness checklist completed

---

## 🔗 References

- Annotation Guideline: `/services/synthesizer_v2/assets/annotation-guideline.md`
- Label Structure: `/services/synthesizer_v2/assets/structure/label_structure.csv`
- Subtype Definitions: `/services/synthesizer_v2/assets/structure/gb_subtype_definition.csv`
- Decision Framework: `/services/synthesizer_v2/assets/structure/decision_framework.csv`

---

**Document Version:** 2.0  
**Last Updated:** April 22, 2026  
**Maintained by:** Gender Bias Detection Team
