#!/usr/bin/env python3
"""
generate_synthetic.py — Generate synthetic gender-bias training data from annotated_dataset.csv

Reads annotated_dataset.csv (real annotated data), counts each class, and for every class
with fewer than TARGET samples uses Gemini to synthesize more.

Strategy per class:
  1. Sample one real example from the class
  2. Send that example + label + description to Gemini
  3. Gemini returns 5 diverse synthetic sentences with the same label
  4. Repeat (cycling through real examples) until the class reaches TARGET

Output: synthetic_dataset.csv  (text, binary_label, subtype)
  binary_label — GB | NON-GB
  subtype      — GB-ATTACK | GB-NORMATIVE | GB-SEX | NON-GB

Classes and targets (TARGET = 2000 by default):
  GB-ATTACK    122 real  → generate ~1878
  GB-NORMATIVE 290 real  → generate ~1710
  GB-SEX        49 real  → generate ~1951
  NON-GB      2328 real  → already ≥ 2000, skip

Usage:
    python generate_synthetic.py
    python generate_synthetic.py --target 2000 --output synthetic_dataset.csv
    python generate_synthetic.py --dry-run            # preview prompts only
    python generate_synthetic.py --classes GB-ATTACK  # single class
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ANNOTATED_CSV = SCRIPT_DIR.parents[1] / "data" / "annotated_dataset.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "synthetic_dataset.csv"
DEFAULT_TARGET = 2000
DEFAULT_MODEL = "gemini-2.5-flash"

LABEL_DESCRIPTIONS: Dict[str, str] = {
    "GB-ATTACK": (
        "โจมตี ดูหมิ่น ลดคุณค่าเพศอย่างตรงไปตรงมา — "
        "ใช้ภาษา insulting/derogatory/dehumanizing ต่อเพศหญิง เพศชาย หรือ LGBTQ+ "
        "เช่น ด่าเพศด้วยคำหยาบ ใช้ slur เพศ (กะหรี่ ตุ๊ด ลักเพศ) หรือปฏิเสธคุณค่าของกลุ่มเพศ"
    ),
    "GB-NORMATIVE": (
        "เหมารวม กำหนดบทบาท หรือ policing เพศ — "
        "stereotype คุณลักษณะ/ความสามารถตามเพศ กำหนดหน้าที่ทางเพศ (gender role) "
        "หรือตัดสินว่าใครเป็น 'เพศจริง' เช่น 'ผู้หญิงต้องอยู่บ้าน' 'ทอมจริงต้องแมน'"
    ),
    "GB-SEX": (
        "ลดคุณค่าผ่าน sexualized insult หรือ body-based attack — "
        "ใช้อวัยวะเพศหรือพฤติกรรมทางเพศเพื่อเหยียดกลุ่มเพศ วิจารณ์รูปร่างที่ผูกกับเพศ "
        "หรือ sexual performance shame เป้าหมายต้องเป็นกลุ่มเพศ ไม่ใช่บุคคลเดียว"
    ),
    "NON-GB": (
        "ข้อความที่ไม่เข้าเกณฑ์ gender bias — "
        "ไม่มีการโจมตี เหมารวม หรือ objectify เพศ "
        "อาจมีคำเกี่ยวกับเพศแต่ไม่ใช่การเหยียด เช่น meta-commentary วิจารณ์อคติ "
        "หรือด่าบุคคลเดียวโดยไม่โยงกลุ่มเพศ"
    ),
}

BINARY_LABEL: Dict[str, str] = {
    "GB-ATTACK": "GB",
    "GB-NORMATIVE": "GB",
    "GB-SEX": "GB",
    "NON-GB": "NON-GB",
}

ALL_SUBTYPES = ["GB-ATTACK", "GB-NORMATIVE", "GB-SEX", "NON-GB"]

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# Gemini REST client (self-contained, no external dependencies beyond certifi)
# ---------------------------------------------------------------------------


def _call_gemini(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    response_schema: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> str:
    import ssl
    from urllib import error, request as url_request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    gen_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    if response_schema:
        gen_config["responseSchema"] = response_schema

    body: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_config,
    }
    encoded = json.dumps(body).encode("utf-8")

    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    req = url_request.Request(
        url=url,
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    opener = url_request.build_opener(url_request.HTTPSHandler(context=ctx))

    try:
        with opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc}") from exc

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc


# ---------------------------------------------------------------------------
# Load real annotated data
# ---------------------------------------------------------------------------


def load_real_data(csv_path: Path) -> Dict[str, List[str]]:
    """Return {subtype: [text, ...]} from annotated_dataset.csv.

    NON-GB uses the binary label (all 2328 non-GB rows).
    GB subtypes use the first subtype from wa_label_type.
    """
    examples: Dict[str, List[str]] = defaultdict(list)

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            binary = row["wa_binary"].strip()
            if not text:
                continue

            if binary == "NON-GB":
                examples["NON-GB"].append(text)
                continue

            label_type_raw = row.get("wa_label_type", "").strip()
            if not label_type_raw or label_type_raw == "[]":
                continue
            try:
                subtypes: List[str] = json.loads(label_type_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            # Assign to the first recognized GB subtype
            for st in subtypes:
                if st in ("GB-ATTACK", "GB-NORMATIVE", "GB-SEX"):
                    examples[st].append(text)
                    break

    return dict(examples)


def print_class_counts(real_data: Dict[str, List[str]], target: int) -> None:
    print(f"\n{'Class':<20} {'Real':>8} {'Need':>8}  Status")
    print("-" * 55)
    for subtype in ALL_SUBTYPES:
        count = len(real_data.get(subtype, []))
        need = max(0, target - count)
        status = "skip (already ≥ target)" if need == 0 else f"generate {need:,}"
        print(f"{subtype:<20} {count:>8,} {need:>8,}  {status}")
    print()


# ---------------------------------------------------------------------------
# Prompts — one system prompt per class, grounded in annotation-guideline.md
# ---------------------------------------------------------------------------

_SYSTEM_GB_ATTACK = """\
คุณคือผู้ช่วยวิจัย NLP สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับฝึก text classifier ตรวจจับอคติทางเพศ
คลาสที่ต้องสร้าง: GB-ATTACK

━━ นิยาม ━━
ข้อความที่โจมตี ดูหมิ่น ลดคุณค่า หรือเหยียดเพศอย่างตรงไปตรงมา
โดยใช้ภาษา insulting / derogatory / dehumanizing ต่อกลุ่มเพศ

━━ ต้องผ่านเกณฑ์ทั้งสอง ━━
(A) เป้าหมายเป็นกลุ่มเพศ: ผู้หญิง / ผู้ชาย / LGBTQ+ (เกย์ เลส ไบ ทอม ดี้ กะเทย ฯลฯ)
    — อาจอ้างอิงโดยตรง หรืออ้อมผ่านสแลงเพศ เช่น "เฟมทวิต", "ลักเพศ", "ถังขี้", "ประตูหลัง"
(B) มีการดูหมิ่น ลดคุณค่า ปฏิเสธคุณค่า หรือเรียกร้องอันตรายต่อกลุ่มเพศนั้น

━━ ห้ามสร้าง (จะกลายเป็น NON-GB) ━━
✗ (C) Meta/counter-speech: ผู้พูดวิจารณ์อคติของคนอื่น ไม่ได้เหยียดเอง
     ผิด: "ยังมีคนพูดว่าผู้หญิงโง่อยู่เลย น่าสมเพชมาก"
✗ (D) ดูหมิ่นบุคคลเดียว: เจาะจงบุคคล ไม่เหมารวมกลุ่มเพศ
     ผิด: "ผู้หญิงคนนั้นโง่มาก" / "แกนะเกย์งี่เง่า"

━━ รูปแบบ GB-ATTACK ที่ต้องการ ━━
• ด่าเพศโดยตรงด้วยคำหยาบหรือ slur เพศ
  คำ slur ที่ใช้ได้: กะหรี่, ตุ๊ด, อีตุ๊ด, ลักเพศ, ผิดเพศ, อีตัว, ถังขี้, ประตูหลัง, พวกเบี่ยงเบน
• ลดคุณค่าความเป็นมนุษย์เพราะเพศ (dehumanizing)
• ประณามหรือเรียกร้องให้ได้รับอันตรายเพราะเพศ
• เหยียดเพศสภาพ (trans/ทอม/ดี้/กะเทย) โดยตรง
หมายเหตุ: ไม่ต้องเป็น stereotype หรือ sexualized — โจมตีเพศตรงๆ ก็เป็น GB-ATTACK แล้ว

━━ ตัวอย่างที่ถูกต้อง ✓ ━━
"ผู้หญิงมันโง่หมดทุกคน"
"ไอ้ตุ๊ดพวกนี้ไม่นับเป็นคน"
"กะเทยคือของเสียสังคม ควรไปอยู่ที่อื่น"
"ผู้หญิงพวกนี้ไร้ค่าไร้สมอง มีแต่ปาก"
"ผู้หญิงแบบนี้สมควรโดนทำร้าย"
"อีพวกลักเพศไม่ควรมีสิทธิ์แบบคนปกติ"

━━ กฎการสร้าง ━━
- ข้อความ 1–3 ประโยค เหมือน tweet / comment จริง
- ใช้ภาษาโซเชียลมีเดียไทย ลำลอง มีสแลงหยาบได้
- กระจายเป้าหมาย: ผู้หญิง / ผู้ชาย / เกย์ / กะเทย / ทอม / ดี้ / เลส / ไบ ฯลฯ
- แต่ละข้อต้องต่างกันอย่างชัดเจนทั้งโครงสร้าง เนื้อหา บริบท
- ห้ามคัดลอกหรือ paraphrase ตรงๆ จากตัวอย่างที่ให้มา
"""

_SYSTEM_GB_NORMATIVE = """\
คุณคือผู้ช่วยวิจัย NLP สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับฝึก text classifier ตรวจจับอคติทางเพศ
คลาสที่ต้องสร้าง: GB-NORMATIVE

━━ นิยาม ━━
ข้อความที่แสดงอคติจาก "ค่านิยมทางเพศ" ได้แก่
• Stereotype: เหมารวมว่ากลุ่มเพศมีคุณลักษณะ/ความสามารถ X
• Gender role: บอกว่ากลุ่มเพศต้องทำ/ห้ามทำ Y เพราะเพศ
• Identity policing: ตัดสินว่าใครแสดงออก "ถูก" ตามเพศหรือไม่
• Benevolent stereotype: ยกย่องแต่กำหนดบทบาทตายตัวให้กลุ่มเพศ

━━ ต้องผ่านเกณฑ์ทั้งสอง ━━
(A) เป้าหมายเป็นกลุ่มเพศ ไม่ใช่บุคคลเดียว
(B) มีการเหมารวม กำหนดบทบาท หรือ policing กลุ่มเพศ
    — ครอบคลุมเพศหญิง/ชาย และ LGBTQ+

━━ ห้ามสร้าง (จะกลายเป็น NON-GB) ━━
✗ (C) Meta/critique/counter: ผู้พูดวิจารณ์ค่านิยม ไม่ได้บังคับใช้เอง
     ผิด: "ทำไมสังคมยังคิดว่าผู้หญิงควรอยู่บ้าน"
     ผิด: "ใครบอกว่าผู้ชายห้ามร้องไห้ มันคือ toxic masculinity"
✗ (D) ดูหมิ่นบุคคลเดียว: เจาะจงคน ไม่เหมารวมกลุ่ม
     ผิด: "แกนะ ผู้หญิงขี้บ่นมาก หยุดได้แล้ว"

━━ 3 รูปแบบย่อยที่ต้องการ (กระจาย ~33% ต่อประเภท) ━━
1. Stereotype — เหมารวมคุณลักษณะ ความสามารถ อารมณ์ ตามเพศ
   ตัวอย่าง: "ผู้หญิงดูบอลไม่เป็นหรอก ไม่ต้องมาแสร้งทำ"
             "กะเทยเป็นคนขี้เม้าท์ทุกคน พูดเยอะเกินจริง"
             "เกย์พวกนี้อ่อนแอหมด ทำงานหนักไม่ไหว"

2. Gender Role — กำหนดสิ่งที่เพศต้องทำ/ห้ามทำ หน้าที่ตามเพศ
   ตัวอย่าง: "ผู้ชายต้องเป็นผู้นำเสมอ นั่นคือธรรมชาติ"
             "ผู้หญิงไม่ควรทำงานหนัก ปล่อยให้ผู้ชายทำ"
             "ผู้ชายที่อยู่บ้านเลี้ยงลูกไม่ใช่ผู้ชายจริงๆ"

3. Identity Policing — ตัดสินว่าใครเป็น "เพศจริง" หรือแสดงออกถูกต้อง
   ตัวอย่าง: "ทอมที่แท้จริงต้องแมนกว่านี้ อย่ามาทำอ่อนแอ"
             "เกย์จริงต้องมีนิสัยผู้หญิง ไม่งั้นก็ไม่ใช่เกย์"
             "ชายแท้ต้องไม่ร้องไห้ ร้องไห้แล้วใครจะเคารพ"

━━ กฎการสร้าง ━━
- ข้อความ 1–3 ประโยค เหมือน tweet / comment จริง
- อาจสุภาพหรือหยาบก็ได้ แต่ต้องมีนัยเหมารวม/กำหนดบทบาทกลุ่มเพศ
- กระจายเป้าหมาย: ผู้หญิง / ผู้ชาย / ทอม / ดี้ / เกย์ / กะเทย / เลส ฯลฯ
- แต่ละข้อต้องต่างกันอย่างชัดเจน
- ห้ามคัดลอกหรือ paraphrase ตรงๆ จากตัวอย่างที่ให้มา
"""

_SYSTEM_GB_SEX = """\
คุณคือผู้ช่วยวิจัย NLP สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับฝึก text classifier ตรวจจับอคติทางเพศ
คลาสที่ต้องสร้าง: GB-SEX

━━ นิยาม ━━
ข้อความที่ลดคุณค่าหรือดูหมิ่นเพศผ่านเนื้อหา sexualized หรือ body-based
เป้าหมายต้องเป็น "กลุ่มเพศ" เสมอ — ไม่ใช่บุคคลเดียว

━━ ต้องผ่านเกณฑ์ทั้งสอง ━━
(A) เป้าหมายเป็นกลุ่มเพศ (ไม่ใช่บุคคลเดียว)
    — ต้องพูดถึงกลุ่ม เช่น "ผู้ชายพวกนี้", "กะเทยทั้งหลาย", "ผู้หญิงแบบนี้"
(B) ใช้ร่างกาย / อวัยวะเพศ / สมรรถภาพทางเพศ เพื่อลดคุณค่ากลุ่มเพศ

━━ ห้ามสร้าง (จะกลายเป็น NON-GB) ━━
✗ (C) Meta/critique: ผู้พูดวิจารณ์อคติทางเพศ ไม่ได้เหยียดเอง
✗ (D) ด่าบุคคลเดียว: ระบุตัวบุคคลชัดเจน ไม่ได้เหมารวมกลุ่มเพศ
     ผิด: "มึงหีดำมาก" (บุคคลเดียว → NON-GB)
     ผิด: "แกนี่ควยเล็กมากเลย" (บุคคลเดียว → NON-GB)

━━ 3 รูปแบบย่อยที่ต้องการ (กระจาย ~33% ต่อประเภท) ━━
1. Sexualized attack ต่อกลุ่มเพศ — ใช้อวัยวะเพศหรือ sexual role ดูหมิ่นกลุ่มเพศ
   คำที่ใช้ได้: หี, ควย, จู๋, ตูด, นม, อวัยวะเพศ ฯลฯ (ในบริบทดูหมิ่นกลุ่ม)
   ตัวอย่าง: "หีดำแบบนี้มันไม่ใช่ผู้หญิงดีๆ"
             "ควยเล็กแล้วจะเป็นผู้ชายได้ไง"
             "ตุ๊ดพวกนี้ใช้รูตูดอย่างเดียว คุ้มค่าดี"

2. Body-shaming ที่ผูกกับกลุ่มเพศ — วิจารณ์รูปร่างที่ผูกกับเพศในเชิงลบ
   ตัวอย่าง: "กะเทยนมปลอมเหมือนบอลลูน ดูน่าขยะแขยง"
             "ผู้หญิงอ้วนพวกนี้ขายตัวไม่ได้หรอก ใครจะเอา"
             "กะเทยนมยาน อย่ามาอวดเลย"

3. Sexual performance shame ต่อกลุ่มเพศ — ตอกย้ำว่าคุณค่าขึ้นกับสมรรถภาพทางเพศ
   ตัวอย่าง: "ผู้ชายแบบนี้คงไร้น้ำยา"
             "ก็จู๋เล็กทั้งนั้น เลยต้องทำตัวกร่าง แก้ปม"
             "ผู้ชายไม่มีน้ำยา อยู่ได้ยังไงวะ"

━━ จุดแยก GB-SEX จาก GB-ATTACK ━━
GB-ATTACK: ด่าตรงๆ ใช้ slur เพศ ไม่จำเป็นต้องเกี่ยวกับร่างกาย/เพศทางกายภาพ
GB-SEX: ต้องใช้ร่างกาย อวัยวะเพศ หรือ sexual performance เป็นเครื่องมือดูหมิ่น

━━ กฎการสร้าง ━━
- ข้อความ 1–3 ประโยค เหมือน tweet / comment จริง
- ภาษาอาจหยาบคาย อวัยวะเพศอาจปรากฏในบริบทดูหมิ่นกลุ่ม
- กระจายเป้าหมาย: ผู้หญิง / ผู้ชาย / กะเทย / เกย์ ฯลฯ
- กระจายทั้ง 3 รูปแบบย่อย
- ห้ามคัดลอกหรือ paraphrase ตรงๆ จากตัวอย่างที่ให้มา
"""

_SYSTEM_NON_GB = """\
คุณคือผู้ช่วยวิจัย NLP สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับฝึก text classifier ตรวจจับอคติทางเพศ
คลาสที่ต้องสร้าง: NON-GB

━━ นิยาม ━━
ข้อความที่ไม่เข้าเกณฑ์ Gender Bias — ไม่มีการโจมตี เหมารวม หรือ objectify กลุ่มเพศ

━━ NON-GB ครอบคลุม 4 รูปแบบ ━━
1. Neutral: ข้อความทั่วไป ไม่มีเนื้อหาเพศเลย
   "วันนี้อากาศร้อนมาก" | "ราคาข้าวของแพงขึ้น" | "รถติดมากเลย"

2. Gendered Insult (D): ด่าบุคคลเดียวโดยมีการกล่าวถึงเพศ แต่ไม่เหมารวมกลุ่ม
   เงื่อนไข: มีคำระบุตัวบุคคล (คนนั้น/แก/มึง/ยาย/ไอ้นี่) + เพศ + คำลบ
   "คนนั้นอะ ผู้ชายโง่ๆ" | "ผู้หญิงคนนั้นขี้วีนมาก" | "เกย์คนนั้นชอบเถียง"
   ห้าม: ใช้ slur เพศ (กะหรี่ ตุ๊ด ลักเพศ) เพราะนั่นจะเป็น GB-ATTACK

3. Meta/Counter-speech (C): วิจารณ์อคติหรือ social critique
   "ยังมีคนพูดว่าผู้หญิงโง่อยู่ น่าสมเพช"
   "สังคมไทยชอบโทษผู้หญิงเวลาโดนคุกคาม"
   "ใครบอกว่าผู้ชายห้ามร้องไห้ มันคือ toxic masculinity"

4. Sexual Content ที่ไม่ใช่ bias: เนื้อหาทางเพศที่ไม่ได้เหยียดกลุ่มเพศ
   "เมื่อคืนมีอะไรกับแฟน" | "อยากอมควยใจจะขาด" (เป็นบุคคลเดียว ไม่เหมารวม)

━━ ห้ามสร้าง ━━
✗ ข้อความที่เหมารวม ด่า หรือกำหนดบทบาทกลุ่มเพศ (นั่นคือ GB)
✗ ใช้ slur เพศ (กะหรี่ ตุ๊ด ลักเพศ ฯลฯ) แม้จะด่าบุคคลเดียว

━━ กฎการสร้าง ━━
- ข้อความ 1–3 ประโยค เหมือน tweet / comment จริง
- กระจายทั้ง 4 รูปแบบ (~25% ต่อประเภท)
- หัวข้อหลากหลาย: การเมือง กีฬา อาหาร จราจร เงิน งาน ดาราบันเทิง
- ห้ามคัดลอกหรือ paraphrase ตรงๆ จากตัวอย่างที่ให้มา
"""

_CLASS_SYSTEMS: Dict[str, str] = {
    "GB-ATTACK": _SYSTEM_GB_ATTACK,
    "GB-NORMATIVE": _SYSTEM_GB_NORMATIVE,
    "GB-SEX": _SYSTEM_GB_SEX,
    "NON-GB": _SYSTEM_NON_GB,
}

# ---------------------------------------------------------------------------
# Rotation tables — keep sub-pattern and target balanced across API calls
# ---------------------------------------------------------------------------

_CLASS_ROTATIONS: Dict[str, Dict[str, List[str]]] = {
    "GB-ATTACK": {
        "targets": [
            "ผู้หญิง / กลุ่มผู้หญิง",
            "ผู้ชาย / กลุ่มผู้ชาย",
            "เกย์ / กลุ่มชายรักชาย",
            "กะเทย / สาวประเภทสอง",
            "ทอม / ดี้",
            "เลสเบี้ยน / หญิงรักหญิง",
            "ไบเซ็กชวล",
            "LGBTQ+ โดยรวม",
        ],
        "subpatterns": [
            "ด่าเพศโดยตรงด้วยคำหยาบหรือ slur (กะหรี่, ตุ๊ด, ลักเพศ, ผิดเพศ ฯลฯ)",
            "ลดคุณค่าความเป็นมนุษย์ (dehumanizing) เพราะเพศ",
            "ประณามหรือเรียกร้องให้ได้รับอันตรายเพราะเพศ",
            "ปฏิเสธคุณค่าหรือสิทธิ์ของกลุ่มเพศ",
        ],
    },
    "GB-NORMATIVE": {
        "targets": [
            "ผู้หญิง / กลุ่มผู้หญิง",
            "ผู้ชาย / กลุ่มผู้ชาย",
            "ทอม",
            "ดี้",
            "เกย์",
            "กะเทย / สาวประเภทสอง",
            "เลสเบี้ยน",
        ],
        "subpatterns": [
            "stereotype — เหมารวมคุณลักษณะ ความสามารถ หรืออารมณ์ตามเพศ",
            "gender role — กำหนดหน้าที่หรือสิ่งที่เพศต้องทำ / ห้ามทำ",
            "identity policing — ตัดสินว่าใครแสดงออกตามเพศ 'ถูกต้อง' หรือไม่",
        ],
    },
    "GB-SEX": {
        "targets": [
            "ผู้หญิง / กลุ่มผู้หญิง",
            "ผู้ชาย / กลุ่มผู้ชาย",
            "กะเทย / สาวประเภทสอง",
            "เกย์ / ชายรักชาย",
            "ทอม / ดี้",
            "เลสเบี้ยน / หญิงรักหญิง",
        ],
        "subpatterns": [
            "sexualized attack — ใช้อวัยวะเพศหรือ sexual role ดูหมิ่นกลุ่มเพศ",
            "body-shaming ที่ผูกกับเพศ — วิจารณ์รูปร่าง/ขนาดร่างกายที่ผูกกับเพศในเชิงลบ",
            "sexual performance shame — ตอกย้ำว่าคุณค่าขึ้นกับสมรรถภาพทางเพศ",
        ],
    },
    "NON-GB": {
        "targets": ["ไม่ระบุเพศ / ทั่วไป"],
        "subpatterns": [
            "neutral — ข้อความทั่วไปไม่มีเนื้อหาเพศ (การเมือง กีฬา อาหาร จราจร งาน ฯลฯ)",
            "gendered insult (D) — ด่าบุคคลเดียวโดยมีการกล่าวถึงเพศ แต่ไม่เหมารวมกลุ่ม",
            "meta / counter-speech (C) — วิจารณ์อคติทางเพศ / social critique",
            "sexual content ที่ไม่ใช่ bias — เนื้อหาทางเพศที่ไม่ได้เหยียดกลุ่มเพศ",
        ],
    },
}

_USER_TEMPLATE = """\
ตัวอย่างข้อความจริงจาก dataset (ใช้เป็น seed เท่านั้น อย่าคัดลอก):
"{text}"

คลาส: {subtype}
เป้าหมายรอบนี้: {target}
รูปแบบย่อยรอบนี้: {subpattern}

สร้างข้อความสังเคราะห์ 5 ประโยคที่เป็นคลาส {subtype}
- ทั้ง 5 ข้อต้องยึดเกณฑ์ใน system prompt
- เน้นเป้าหมาย "{target}" และรูปแบบ "{subpattern}" เป็นหลัก
- ห้ามขึ้นต้นซ้ำกัน ต้องต่างโครงสร้างและบริบทกันทุกข้อ
"""


def _make_response_schema(subtype: str) -> Dict[str, Any]:
    binary = BINARY_LABEL[subtype]
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "binary_label": {"type": "string", "enum": [binary]},
                        "subtype": {"type": "string", "enum": [subtype]},
                    },
                    "required": ["text", "binary_label", "subtype"],
                },
            }
        },
        "required": ["items"],
    }


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: str) -> List[Dict[str, Any]]:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        for key in ("items", "data", "results", "sentences"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        return []
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


# ---------------------------------------------------------------------------
# Generate synthetic data for one class
# ---------------------------------------------------------------------------


def generate_for_class(
    *,
    subtype: str,
    real_examples: List[str],
    needed: int,
    api_key: str,
    model: str,
    temperature: float,
    dry_run: bool,
    checkpoint_path: Path,
    retry_delay: float = 5.0,
) -> List[Dict[str, Any]]:
    if not real_examples:
        print(f"  [warning] No real examples found for {subtype}, skipping.")
        return []

    system_prompt = _CLASS_SYSTEMS[subtype]
    schema = _make_response_schema(subtype)
    rotations = _CLASS_ROTATIONS[subtype]
    rot_targets = rotations["targets"]
    rot_subpatterns = rotations["subpatterns"]

    # Resume from checkpoint
    generated: List[Dict[str, Any]] = [] if dry_run else _load_checkpoint(checkpoint_path)
    if generated:
        print(f"  [resume] {len(generated)} already generated, continuing from checkpoint")

    seen_texts = {item["text"] for item in generated}
    # Near-duplicate guard: reject items that share their first 30 chars with an existing item
    seen_prefixes = {t[:30] for t in seen_texts}
    remaining = needed - len(generated)

    if remaining <= 0:
        print(f"  [skip] checkpoint already has {len(generated)}/{needed} items")
        return generated[:needed]

    # Cycle through real examples as seeds (shuffle each cycle)
    pool = list(real_examples)
    random.shuffle(pool)
    pool_idx = 0
    call_idx = 0          # drives rotation independently of seed cycling
    consecutive_errors = 0

    while remaining > 0:
        if pool_idx >= len(pool):
            random.shuffle(pool)
            pool_idx = 0

        seed_text = pool[pool_idx]
        pool_idx += 1

        # Rotate target and sub-pattern to keep distribution balanced
        target = rot_targets[call_idx % len(rot_targets)]
        subpattern = rot_subpatterns[call_idx % len(rot_subpatterns)]
        call_idx += 1

        user_prompt = _USER_TEMPLATE.format(
            text=seed_text,
            subtype=subtype,
            target=target,
            subpattern=subpattern,
        )

        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"DRY RUN — {subtype}")
            print(f"  seed   : {seed_text[:80]}")
            print(f"  target : {target}")
            print(f"  subpat : {subpattern}")
            print(f"{'=' * 60}")
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("\nUSER PROMPT:")
            print(user_prompt)
            return []

        try:
            raw = _call_gemini(
                api_key=api_key,
                model=model,
                system=system_prompt,
                user=user_prompt,
                temperature=temperature,
                max_tokens=4096,
                response_schema=schema,
            )
            consecutive_errors = 0

            items = _parse_response(raw)
            added = 0
            for item in items:
                if remaining <= 0:
                    break
                text = item.get("text", "").strip()
                if not text:
                    continue
                if text in seen_texts:
                    continue
                if item.get("subtype") != subtype:
                    continue
                # Near-duplicate guard: skip if opening 30 chars already seen
                prefix = text[:30]
                if prefix in seen_prefixes:
                    continue
                generated.append(
                    {
                        "text": text,
                        "binary_label": BINARY_LABEL[subtype],
                        "subtype": subtype,
                    }
                )
                seen_texts.add(text)
                seen_prefixes.add(prefix)
                remaining -= 1
                added += 1

            total_so_far = needed - remaining
            print(
                f"  [{subtype}] {total_so_far:>5}/{needed}  (+{added} from this seed, {remaining} remaining)",
                flush=True,
            )
            _save_checkpoint(generated, checkpoint_path)

            if remaining > 0:
                time.sleep(0.3)

        except RuntimeError as exc:
            consecutive_errors += 1
            msg = str(exc)
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                wait = retry_delay * (2 ** min(consecutive_errors, 4))
                print(f"\n  [rate-limit] waiting {wait:.0f}s …", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [error] {exc}", flush=True)
                if consecutive_errors >= 5:
                    print(f"  [abort] too many consecutive errors for {subtype}")
                    break
                time.sleep(retry_delay * consecutive_errors)

    return generated[:needed]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Thai gender-bias training data from annotated_dataset.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Target samples per class (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.95,
        help="Sampling temperature (default: 0.95)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (falls back to GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one prompt per class, no API calls or file writes",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=ALL_SUBTYPES,
        metavar="CLASS",
        help=(f"Only generate for these classes. Choices: {ALL_SUBTYPES}. Default: all classes below --target."),
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load .env
    _load_env(SCRIPT_DIR / ".env")
    _load_env(Path.cwd() / ".env")

    api_key: str = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: GEMINI_API_KEY not set. Add it to .env or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    # Load real data
    if not ANNOTATED_CSV.exists():
        print(f"Error: {ANNOTATED_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {ANNOTATED_CSV} …")
    real_data = load_real_data(ANNOTATED_CSV)
    print_class_counts(real_data, args.target)

    # Decide which classes to generate
    classes_to_run: List[tuple[str, int]] = []
    for subtype in ALL_SUBTYPES:
        if args.classes and subtype not in args.classes:
            continue
        count = len(real_data.get(subtype, []))
        if count < args.target:
            classes_to_run.append((subtype, args.target - count))

    if not classes_to_run:
        print(f"All requested classes already have ≥ {args.target} samples. Nothing to do.")
        return

    output_path = Path(args.output)
    checkpoint_dir = output_path.parent / ".synth_checkpoints"

    all_synthetic: List[Dict[str, Any]] = []

    for subtype, needed in classes_to_run:
        print(f"\n{'─' * 60}")
        print(f"Generating {needed:,} synthetic examples for {subtype}")
        print(f"  Real examples available: {len(real_data.get(subtype, []))}")
        print(f"  Cycles needed: ~{needed // (len(real_data.get(subtype, [])) * 5) + 1} passes through real data")
        print(f"{'─' * 60}")

        ckpt = checkpoint_dir / f"{subtype}.json"
        items = generate_for_class(
            subtype=subtype,
            real_examples=real_data.get(subtype, []),
            needed=needed,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            dry_run=args.dry_run,
            checkpoint_path=ckpt,
        )
        all_synthetic.extend(items)
        print(f"  [{subtype}] complete: {len(items)} synthetic sentences added")

    if args.dry_run:
        print("\n[dry-run] No output written.")
        return

    # Write output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "binary_label", "subtype"])
        writer.writeheader()
        writer.writerows(all_synthetic)

    # Clean checkpoints on success
    for subtype, _ in classes_to_run:
        ckpt = checkpoint_dir / f"{subtype}.json"
        if ckpt.exists():
            ckpt.unlink()

    # Final summary
    counts = Counter(item["subtype"] for item in all_synthetic)
    binary_counts = Counter(item["binary_label"] for item in all_synthetic)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(all_synthetic):,} synthetic items → {output_path}")
    print("\nSynthetic distribution by subtype:")
    for subtype in ALL_SUBTYPES:
        n = counts.get(subtype, 0)
        real_n = len(real_data.get(subtype, []))
        total = real_n + n
        bar = "█" * min(40, total // 50)
        print(f"  {subtype:<20} {real_n:>6} real + {n:>6} synth = {total:>6} total  {bar}")

    print("\nSynthetic distribution by binary label:")
    for label, n in binary_counts.items():
        print(f"  {label:<10} {n:,}")
    print()


if __name__ == "__main__":
    main()
