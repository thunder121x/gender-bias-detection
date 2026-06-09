#!/usr/bin/env python3
"""
Thai Gender Bias Span Detector - Fine-tuning Pipeline for Qwen 3.5 2B
Training on RTX 6000 Blackwell with LoRA (16-bit)

This script converts annotated JSON data into ChatML JSONL format and fine-tunes
Qwen 3.5 2B using Unsloth for efficient training on a single GPU.
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

SERVICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVICE_DIR.parents[1]

# ============================================================================
# PRODUCTION SYSTEM PROMPT (Section 1 - BYTE IDENTICAL)
# ============================================================================

SYSTEM_PROMPT = """คุณคือเครื่องมือตรวจจับช่วงข้อความที่มีอคติทางเพศในภาษาไทย (Thai Gender Bias Span Detector)

หน้าที่ของคุณ:
1. รับข้อความต้นฉบับภาษาไทย
2. คัดลอกข้อความต้นฉบับทุกตัวอักษรอย่างไม่ขาดไม่เกิน (รักษาช่องว่าง เครื่องหมายวรรคตอน อิโมจิ การสะกด ทุกอย่างให้เหมือนต้นฉบับ)
3. ใส่แท็กรอบช่วงข้อความ (span) ที่เข้าเกณฑ์อคติทางเพศตามนิยามด้านล่างเท่านั้น
4. ห้ามอธิบาย ห้ามตีความ ห้ามแก้ไข ห้ามแปล ห้ามเพิ่มหรือลบเนื้อหาใด ๆ นอกเหนือจากการใส่แท็ก

# แท็กที่อนุญาต (มีเพียง 3 ประเภทเท่านั้น)

<GB-ATTACK>...</GB-ATTACK>
- การโจมตี ดูหมิ่น ลดทอนความเป็นมนุษย์ ของกลุ่มเพศโดยตรง (เพศหญิง เพศชาย และ LGBTQ+)
- ใช้ภาษา insulting / derogatory / dehumanizing
- รวมถึง "คำเหยียดเพศโดยเนื้อหา" (inherently gendered slurs) เช่น กะหรี่ ร่าน อีตัว สำส่อน อีตุ๊ด ลักเพศ ผิดเพศ ถังขี้ ประตูหลัง — แม้จะด่าบุคคลคนเดียว ให้ใส่แท็กเพราะคำเหล่านี้ลดคุณค่ากลุ่มเพศโดยเนื้อหาของคำเอง
- ตัวอย่าง: "ผู้หญิงมันโง่หมดทุกคน", "ไอ้ตุ๊ดพวกนี้ไม่นับเป็นคน", "กะเทยคือของเสียสังคม"

<GB-NORMATIVE>...</GB-NORMATIVE>
- การเหมารวม (stereotype) กำหนดบทบาททางเพศ (gender role) บังคับหรือตรวจสอบอัตลักษณ์ทางเพศ (identity policing)
- รวมถึง benevolent / positive stereotype ที่ผูกคุณลักษณะตายตัวกับกลุ่มเพศ
- ตัวอย่าง: "ผู้ชายต้องเป็นผู้นำเสมอ", "ทอมที่แท้จริงต้องแมนกว่านี้", "ผู้หญิงโดยธรรมชาติอ่อนโยน"

<GB-SEX>...</GB-SEX>
- การลดคุณค่ากลุ่มเพศผ่านอวัยวะเพศ ร่างกายที่ผูกกับเพศ สมรรถภาพทางเพศ หรือ sexualized insult ที่โจมตีกลุ่มเพศ
- รวมถึง body-shaming ที่อ้างอิงเพศ และ sexual performance shame ที่ตอกย้ำคุณค่ากลุ่มเพศ
- ตัวอย่าง: "หีดำแบบนี้มันไม่ใช่ผู้หญิงดี ๆ", "ควยเล็กแล้วจะเป็นผู้ชายได้ไง", "กะเทยนมปลอมเหมือนบอลลูน"

# กฎการตัดสินใจ (Decision Rules)

ใส่แท็กก็ต่อเมื่อข้อความผ่านเงื่อนไข **ทั้งหมด** ดังนี้:
(A) มีเป้าหมายเป็น "กลุ่มเพศ" (เพศ/เพศสภาพ/เพศวิถี) ทั้งโดยตรง (ผู้หญิง ผู้ชาย เกย์ ทอม ดี้ กะเทย ฯลฯ) หรือโดยอ้อม (เฟมทวิต สาววาย ถั่วดำ ตีฉิ่ง พ่อบ้านใจกล้า ฯลฯ)
(B) มีการประเมินค่าลบ ดูหมิ่น เหมารวม กำหนดบทบาท ผูกคุณค่ากับร่างกาย/สมรรถภาพทางเพศ หรือ benevolent stereotype ต่อกลุ่มเพศนั้น
(C) ผู้พูดต้องเป็นผู้แสดงอคติเอง — ไม่ใช่การวิจารณ์ การโต้แย้ง หรือการอ้างถึงอคติของคนอื่น (meta / counter-speech)
(D) ไม่ใช่การด่าบุคคลคนเดียวด้วยคำดูหมิ่นทั่วไปที่ไม่ใช่ slur เพศโดยเนื้อหา

ข้อยกเว้นที่ต้องใส่แท็กแม้เป้าหมายเป็นบุคคลเดียว:
- เมื่อใช้ "คำเหยียดเพศโดยเนื้อหา" (Woman-Degrading Slurs / Anti-LGBT Slurs / Gendered Sexualized Slurs) ให้ถือว่าเข้าเกณฑ์ (A) และ (B) อัตโนมัติ และใส่แท็กตามประเภทของ slur นั้น

# กรณีที่ห้ามใส่แท็กใด ๆ (ส่งคืนข้อความเดิมทุกตัวอักษร)

- Neutral: ไม่กล่าวถึงเพศ เช่น "วันนี้อากาศดีมาก"
- Gendered Insult ต่อบุคคลคนเดียว ที่ไม่ใช้ slur เพศ เช่น "ผู้ชายคนนั้นน่ารำคาญ", "ผู้หญิงคนนั้นพูดจาไม่ดีเลย"
- Meta commentary / Social critique / Counter-speech เช่น "ยังมีคนพูดว่าผู้หญิงโง่อยู่เลย น่าสมเพชมาก", "สังคมไทยชอบโทษผู้หญิงเวลาโดนคุกคามทางเพศ"
- เนื้อหาทางเพศปกติที่ไม่เหยียดกลุ่มเพศ เช่น "เมื่อคืนมีอะไรกับแฟน", "อยากอมควยใจจะขาดเลยดิ"
- อคติประเภทอื่นที่ไม่เกี่ยวกับเพศ (การเมือง ศาสนา ชนชั้น เชื้อชาติ)
- การอ้างถึงเพศในเชิงข้อมูล/ระบุตัวตน เช่น "เราเป็นผู้หญิงค่ะ"

# วิธีเลือกช่วงข้อความ (Span Selection)

- เลือก minimal span ที่ครอบคลุมองค์ประกอบของอคติให้ครบ — ทั้งคำอ้างอิงเพศและคำที่แสดงการเหยียด/เหมารวม/sexualized
- หากในข้อความเดียวมีอคติหลายช่วง ใส่หลายแท็กได้
- ห้ามครอบคลุมข้อความที่ไม่เกี่ยวข้องกับอคติ
- ห้ามแท็กซ้อนแท็ก (no nesting)

# รูปแบบผลลัพธ์ (Output Format)

- คัดลอกข้อความต้นฉบับทั้งหมดให้เหมือนเดิมทุกตัวอักษร
- เพิ่มเฉพาะแท็กที่อนุญาต 3 ประเภทเท่านั้น
- ห้ามมีคำอธิบาย ห้ามมีหมายเหตุ ห้ามมี markdown หรือ JSON หรือเครื่องหมายอื่นใดนอกแท็ก
- หากไม่พบอคติทางเพศตามเกณฑ์ ให้ส่งคืนข้อความต้นฉบับทันทีโดยไม่มีแท็กใด ๆ"""

# ============================================================================
# FEW-SHOT EXAMPLES (Section 2)
# ============================================================================

FEW_SHOT_EXAMPLES = [
    {"user": "ผู้หญิงสมัยนี้สมองกลวงทุกคน", "assistant": "<GB-ATTACK>ผู้หญิงสมัยนี้สมองกลวงทุกคน</GB-ATTACK>"},
    {"user": "ผู้ชายต้องเป็นผู้นำเสมอ ห้ามอ่อนแอ", "assistant": "<GB-NORMATIVE>ผู้ชายต้องเป็นผู้นำเสมอ ห้ามอ่อนแอ</GB-NORMATIVE>"},
    {"user": "ตุ๊ดพวกนี้ใช้รูตูดอย่างเดียว", "assistant": "<GB-SEX>ตุ๊ดพวกนี้ใช้รูตูดอย่างเดียว</GB-SEX>"},
    {"user": "ผู้ชายคนนั้นเอาแต่ใจตัวเองว่ะ", "assistant": "ผู้ชายคนนั้นเอาแต่ใจตัวเองว่ะ"},
    {"user": "ยังมีคนพูดว่าผู้หญิงขับรถไม่เป็น น่าสมเพชจริง ๆ", "assistant": "ยังมีคนพูดว่าผู้หญิงขับรถไม่เป็น น่าสมเพชจริง ๆ"},
    {"user": "วันนี้อากาศดีมาก ไปเดินเล่นกันเถอะ", "assistant": "วันนี้อากาศดีมาก ไปเดินเล่นกันเถอะ"},
]


# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file (handles both list and newline-delimited)"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try parsing as JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try parsing as newline-delimited JSON
    data = []
    for line in content.split("\n"):
        if line.strip():
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return data


def generate_assistant_output(text: str, subtype: str) -> str:
    """
    For mixed content format: text already contains inline tags.
    Return it as-is.
    """
    return text


def build_chatml_message(untagged_text: str, tagged_text: str, subtype: str) -> Dict[str, Any]:
    """
    Build a single ChatML training message for mixed content.

    User input: untagged text (no tags)
    Assistant output: same text with inline <GB-*> tags around bias spans
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": untagged_text},
            {"role": "assistant", "content": tagged_text},
        ]
    }


def load_asset_data(
    assets_dir: str = str(SERVICE_DIR / "assets"),
) -> Dict[str, List[str]]:
    """
    Load LLM-generated bias and non-bias sentences from asset files.

    Returns: Dict with keys: gb_attack, gb_normative, gb_sex, non_gb
    Each value is a list of sentences
    """
    print(f"[LOAD ASSETS] Loading LLM-generated sentences from {assets_dir}...")

    asset_data = {
        "gb_attack": [],
        "gb_normative": [],
        "gb_sex": [],
        "non_gb": [],
    }

    # Load bias data
    for bias_type in ["gb_attack", "gb_normative", "gb_sex"]:
        file_path = os.path.join(assets_dir, f"{bias_type}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            asset_data[bias_type] = [item["text"] for item in data]
        print(f"  {bias_type}: {len(asset_data[bias_type])} sentences")

    # Load non-bias data (combine all 3 categories)
    for non_gb_type in ["non_gb_insult", "non_gb_meta", "non_gb_neutral"]:
        file_path = os.path.join(assets_dir, f"{non_gb_type}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            asset_data["non_gb"].extend([item["text"] for item in data])

    print(f"  non_gb (combined): {len(asset_data['non_gb'])} sentences")
    print(f"  TOTAL: {sum(len(v) for v in asset_data.values())} sentences")

    return asset_data


def generate_data(
    num_samples: int = 30000,
    random_seed: int = 42,
    assets_dir: str = str(SERVICE_DIR / "assets"),
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Generate training data with MIXED bias and non-bias content using REAL sentences from assets.

    Each example is a realistic article/conversation with 1-20 sentences:
    - Mostly non-bias sentences (60-80%)
    - Scattered bias sentences (20-40%) with INLINE tags
    - Multiple bias types can appear in one example
    - Output has tags ONLY around biased portions

    Example output:
    Input:  "สวัสดี ผู้พี่ต้องต้องเก่งด้านการงาน พวกนี้อีตุ๊ดมันไม่มีคุณค่า"
    Output: "สวัสดี <GB-NORMATIVE>ผู้พี่ต้องต้องเก่งด้านการงาน</GB-NORMATIVE>
             <GB-ATTACK>พวกนี้อีตุ๊ดมันไม่มีคุณค่า</GB-ATTACK>"

    Returns: (data_items, labels) for stratification
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # ========== LOAD REAL DATA FROM ASSETS ==========
    asset_data = load_asset_data(assets_dir)

    # ========== HELPER FUNCTIONS ==========

    def sample_sentence(bias_type: str) -> str:
        """Sample a sentence from the specified bias type"""
        if bias_type == "GB-ATTACK":
            return random.choice(asset_data["gb_attack"])
        elif bias_type == "GB-NORMATIVE":
            return random.choice(asset_data["gb_normative"])
        elif bias_type == "GB-SEX":
            return random.choice(asset_data["gb_sex"])
        else:  # NON-GB
            return random.choice(asset_data["non_gb"])

    # ========== GENERATE MIXED EXAMPLES ==========

    def generate_mixed_article() -> Tuple[str, str, str]:
        """
        Generate one example with mixed bias/non-bias sentences from REAL asset data.

        Returns: (untagged_text, tagged_text, primary_label)
        """
        # 1-20 sentences, weighted toward 3-10 (more natural article length)
        num_sents = random.choices(
            list(range(1, 21)),
            weights=[
                0.02,
                0.10,
                0.10,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.10,  # 1-10
                0.04,
                0.04,
                0.04,
                0.04,
                0.03,
                0.03,
                0.03,
                0.02,
                0.02,
                0.01,
            ],  # 11-20
        )[0]

        # 20-40% bias, 60-80% non-bias
        num_bias = max(1, int(num_sents * random.uniform(0.2, 0.4)))
        num_non_bias = num_sents - num_bias

        sentences = []
        bias_types = []

        # Add non-bias sentences
        for _ in range(num_non_bias):
            sentences.append((sample_sentence("NON-GB"), "NON-GB"))

        # Add bias sentences (mixed types)
        for _ in range(num_bias):
            bias_type = random.choices(["GB-ATTACK", "GB-NORMATIVE", "GB-SEX"], weights=[0.33, 0.33, 0.34])[0]
            sentences.append((sample_sentence(bias_type), bias_type))
            bias_types.append(bias_type)

        # Shuffle to mix bias and non-bias naturally
        random.shuffle(sentences)

        # Build untagged and tagged versions
        untagged_parts = []
        tagged_parts = []

        for text, bias_type in sentences:
            untagged_parts.append(text)
            if bias_type != "NON-GB":
                tagged_parts.append(f"<{bias_type}>{text}</{bias_type}>")
            else:
                tagged_parts.append(text)

        untagged_text = " ".join(untagged_parts)
        tagged_text = " ".join(tagged_parts)

        # Primary label: first bias type or NON-GB
        primary_label = bias_types[0] if bias_types else "NON-GB"

        return untagged_text, tagged_text, primary_label

    # ========== MAIN GENERATION LOOP ==========

    print(f"\n[GENERATE] Creating {num_samples:,} mixed realistic examples...")
    print("           (Using REAL LLM-generated sentences from assets)")
    print("           (1-20 sentences per example, bias + non-bias mixed)")

    data_items = []
    labels = []

    for i in range(num_samples):
        untagged, tagged, primary_label = generate_mixed_article()

        data_items.append({"text": untagged, "tagged_text": tagged, "subtype": primary_label})
        labels.append(primary_label)

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1:,} examples...")

    print(f"  ✅ Total: {len(data_items):,}")

    return data_items, labels


def create_training_dataset(
    source_dir: str = str(REPO_ROOT / "services" / "synthesizer" / "output"),
    output_dir: str = str(SERVICE_DIR / "training_data"),
    val_split: float = 0.05,
) -> Dict[str, str]:
    """
    Create training and validation JSONL files from source JSON data.

    Returns: Dict with paths to train_file and val_file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    print("[1/4] Loading synthesized data...")
    gb_attack = load_json_data(os.path.join(source_dir, "gb_attack.json"))
    gb_normative = load_json_data(os.path.join(source_dir, "gb_normative.json"))
    gb_sex = load_json_data(os.path.join(source_dir, "gb_sex.json"))
    non_gb_insult = load_json_data(os.path.join(source_dir, "non_gb_insult.json"))
    non_gb_meta = load_json_data(os.path.join(source_dir, "non_gb_meta.json"))
    non_gb_neutral = load_json_data(os.path.join(source_dir, "non_gb_neutral.json"))

    print(f"  GB-ATTACK: {len(gb_attack)}")
    print(f"  GB-NORMATIVE: {len(gb_normative)}")
    print(f"  GB-SEX: {len(gb_sex)}")
    print(f"  NON-GB (insult): {len(non_gb_insult)}")
    print(f"  NON-GB (meta): {len(non_gb_meta)}")
    print(f"  NON-GB (neutral): {len(non_gb_neutral)}")

    total = sum(
        [len(gb_attack), len(gb_normative), len(gb_sex), len(non_gb_insult), len(non_gb_meta), len(non_gb_neutral)]
    )
    print(f"  TOTAL: {total}")

    # Build training messages
    print("\n[2/4] Building ChatML messages...")
    all_messages = []

    # Positive examples
    for item in gb_attack:
        msg = build_chatml_message(item["text"], "GB-ATTACK")
        all_messages.append(msg)

    for item in gb_normative:
        msg = build_chatml_message(item["text"], "GB-NORMATIVE")
        all_messages.append(msg)

    for item in gb_sex:
        msg = build_chatml_message(item["text"], "GB-SEX")
        all_messages.append(msg)

    # Negative examples
    for item in non_gb_insult:
        msg = build_chatml_message(item["text"], "NON-GB")
        all_messages.append(msg)

    for item in non_gb_meta:
        msg = build_chatml_message(item["text"], "NON-GB")
        all_messages.append(msg)

    for item in non_gb_neutral:
        msg = build_chatml_message(item["text"], "NON-GB")
        all_messages.append(msg)

    print(f"  Created {len(all_messages)} ChatML messages")

    # Stratified split (5% validation)
    print("\n[3/4] Splitting into train/val (5% validation, stratified)...")

    # Create stratification labels (source file)
    labels = []
    for msg in all_messages:
        content = msg["messages"][2]["content"]  # assistant response
        if content.startswith("<GB-ATTACK>"):
            labels.append(0)
        elif content.startswith("<GB-NORMATIVE>"):
            labels.append(1)
        elif content.startswith("<GB-SEX>"):
            labels.append(2)
        else:
            labels.append(3)  # NON-GB

    train_msgs, val_msgs, _, _ = train_test_split(
        all_messages, labels, test_size=val_split, stratify=labels, random_state=42
    )

    print(f"  Train: {len(train_msgs)}")
    print(f"  Val: {len(val_msgs)}")

    # Write JSONL files
    print("\n[4/4] Writing JSONL files...")

    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for msg in train_msgs:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for msg in val_msgs:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")

    return {"train": train_file, "val": val_file}


def create_training_dataset_from_generated(
    num_samples: int = 30000,
    output_dir: str = str(SERVICE_DIR / "training_data_generated"),
    val_split: float = 0.05,
    assets_dir: str = str(SERVICE_DIR / "assets"),
) -> Dict[str, str]:
    """
    Create training and validation JSONL files from LLM-generated asset data.

    Generates realistic mixed examples with inline bias tags:
    - Each example: 1-20 sentences (20-40% bias, 60-80% non-bias, shuffled)
    - Multiple bias types can appear in one example
    - Output has INLINE tags only around biased spans
    - Uses REAL sentences from asset files (high quality, LLM-generated)

    Args:
        num_samples: Total number of samples to generate (default 30,000)
        output_dir: Directory to save JSONL files
        val_split: Validation split ratio (default 5%)
        assets_dir: Directory with asset JSON files

    Returns: Dict with paths to train_file and val_file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate data using REAL asset sentences
    print("[1/4] Generating data using real LLM-generated sentences from assets...")
    data_items, labels = generate_data(num_samples=num_samples, assets_dir=assets_dir)

    # Build training messages
    print("\n[2/4] Building ChatML messages...")
    all_messages = []

    for item in data_items:
        msg = build_chatml_message(untagged_text=item["text"], tagged_text=item["tagged_text"], subtype=item["subtype"])
        all_messages.append(msg)

    print(f"  Created {len(all_messages)} ChatML messages")

    # Stratified split
    print(f"\n[3/4] Splitting into train/val ({int(val_split * 100)}% validation, stratified)...")

    train_msgs, val_msgs, _, _ = train_test_split(
        all_messages, labels, test_size=val_split, stratify=labels, random_state=42
    )

    print(f"  Train: {len(train_msgs)}")
    print(f"  Val: {len(val_msgs)}")

    # Write JSONL files
    print("\n[4/4] Writing JSONL files...")

    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for msg in train_msgs:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for msg in val_msgs:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")

    return {"train": train_file, "val": val_file}


# ============================================================================
# SANITY CHECKS
# ============================================================================


def sanity_check_dataset(train_file: str, val_file: str, sample_size: int = 10) -> bool:
    """Verify training data integrity for mixed content format"""
    print("\n[SANITY CHECK] Verifying dataset integrity...")

    issues = []

    for file_path, dataset_name in [(train_file, "TRAIN"), (val_file, "VAL")]:
        print(f"\nChecking {dataset_name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    issues.append(f"{dataset_name} line {i}: Invalid JSON - {e}")
                    continue

                user_content = msg["messages"][1]["content"]
                assistant_content = msg["messages"][2]["content"]

                # Check 1: User input should be untagged
                if "<GB-" in user_content:
                    issues.append(f"{dataset_name} line {i}: User input should not have tags")

                # Check 2: If assistant has tags, check they're balanced
                if "<GB-" in assistant_content:
                    # Count opening and closing tags
                    for tag_type in ["ATTACK", "NORMATIVE", "SEX"]:
                        open_count = assistant_content.count(f"<GB-{tag_type}>")
                        close_count = assistant_content.count(f"</GB-{tag_type}>")
                        if open_count != close_count:
                            issues.append(
                                f"{dataset_name} line {i}: Unbalanced <GB-{tag_type}> tags "
                                f"({open_count} open, {close_count} close)"
                            )

                # Check 3: Assistant output should contain most of the user input text
                # (with tags inserted)
                untagged_assistant = assistant_content
                for tag_type in ["ATTACK", "NORMATIVE", "SEX"]:
                    untagged_assistant = untagged_assistant.replace(f"<GB-{tag_type}>", "")
                    untagged_assistant = untagged_assistant.replace(f"</GB-{tag_type}>", "")

                if untagged_assistant.strip() != user_content.strip():
                    issues.append(
                        f"{dataset_name} line {i}: Assistant output without tags doesn't match input "
                        f"(input: {len(user_content)} chars, output: {len(untagged_assistant)} chars)"
                    )

    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues[:5]:  # Show first 5
            print(f"  - {issue}")
        return False

    print("✅ All sanity checks passed!")
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thai Gender Bias Span Detector - Data Preparation")
    parser.add_argument(
        "--mode",
        choices=["real", "generated"],
        default="real",
        help="Data source: 'real' (from files) or 'generated' (synthetic, 30K samples)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=30000, help="Number of samples to generate (only for --mode=generated)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("QWEN 3.5 2B - THAI GENDER BIAS SPAN DETECTOR")
    print("Fine-tuning Data Preparation Pipeline")
    print("=" * 80)

    # Create training dataset
    if args.mode == "generated":
        print(f"\n📊 Using GENERATED DATA ({args.num_samples:,} samples)")
        dataset_paths = create_training_dataset_from_generated(num_samples=args.num_samples)
    else:
        print("\n📁 Using REAL DATA (from files)")
        dataset_paths = create_training_dataset()

    # Run sanity checks
    success = sanity_check_dataset(dataset_paths["train"], dataset_paths["val"])

    if success:
        print("\n" + "=" * 80)
        print("✅ DATA PREPARATION COMPLETE")
        print("=" * 80)
        print(f"\nTrain file: {dataset_paths['train']}")
        print(f"Val file: {dataset_paths['val']}")
        print("\nNext steps:")
        print("1. Run: python finetune_qwen_lora.py")
        print("2. This will fine-tune using Unsloth on RTX 6000 Blackwell")
    else:
        print("\n❌ Data preparation failed sanity checks")
        sys.exit(1)
