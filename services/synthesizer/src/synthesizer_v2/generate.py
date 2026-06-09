"""
generate.py — Synthesizer v2
=============================
Generates synthetic Thai social-media text aligned with annotation-guideline.md.

Label taxonomy (Binary + 3 Subtypes):
  GB-ATTACK    — direct attack / slur / dehumanization targeting a gender group
  GB-NORMATIVE — stereotype, gender-role enforcement, identity policing
  GB-SEX       — sexualized attack, body-based gender insult, sexual shame

  NON-GB subtypes:
    neutral         — no gender content at all (criterion: Neutral)
    meta_counter    — meta commentary / social critique / counter-speech (criterion C)
    gendered_insult — insult at one specific person mentioning gender, no group generalization (criterion D)

Output JSON schema per item:
  GB items:
    { "text": "...", "label": "GB", "subtype": "GB-ATTACK|GB-NORMATIVE|GB-SEX",
      "bias_target": "gender group (e.g. ผู้หญิง, เกย์, กะเทย)" }
  NON-GB items:
    { "text": "...", "label": "NON-GB", "subtype": "neutral|meta_counter|gendered_insult" }

Usage:
  synth-v2 --mode gb-attack --count 500 --output output/gb_attack.json
  synth-v2 --mode gb-normative --count 500 --output output/gb_normative.json
  synth-v2 --mode gb-sex --count 500 --output output/gb_sex.json
  synth-v2 --mode non-gb-neutral --count 500 --output output/non_gb_neutral.json
  synth-v2 --mode non-gb-meta --count 300 --output output/non_gb_meta.json
  synth-v2 --mode non-gb-insult --count 300 --output output/non_gb_insult.json
  synth-v2 --mode gb-attack --dry-run

Gemini Batch API (Vertex AI async, 50% cost discount):
  synth-v2 --mode gb-attack --count 500 --provider gemini-batch \\
           --gcp-project my-project --gcs-bucket my-bucket --gcs-prefix synth-v2
  synth-v2 --mode gb-attack --count 500 --provider gemini-batch \\
           --gcp-project my-project --gcs-bucket my-bucket --dry-run
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from synthesizer_v2.constants import (
    ASSETS_DIR,
    BATCH_SIZE,
    DEFAULT_ENV_FILE,
    DEFAULT_MODEL,
    OUTPUT_DIR,
)
from synthesizer_v2.gemini import create_chat_completion

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
# Prompts
# ---------------------------------------------------------------------------

# ── GB-ATTACK ──────────────────────────────────────────────────────────────

_GB_ATTACK_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียไทยประเภท **GB-ATTACK**

คำจำกัดความ GB-ATTACK:
ข้อความที่โจมตี ดูหมิ่น ลดคุณค่า หรือเหยียดเพศอย่างตรงไปตรงมา โดยใช้ภาษา insulting /
derogatory / dehumanizing ต่อเพศหญิง เพศชาย หรือ LGBTQ+ (เกย์ เลส ทอม ดี้ กะเทย ฯลฯ)

เข้าข่ายเมื่อ:
- ด่าเพศโดยตรงด้วยคำหยาบ หรือลดคุณค่าความเป็นมนุษย์เพราะเพศ
- ใช้คำสแลงเหยียดเพศ เช่น กะหรี่, ตุ๊ด, ลักเพศ, ผิดเพศ, อีตัว, ถังขี้
- ประณาม เรียกร้องให้ได้รับอันตราย หรือปฏิเสธคุณค่าของกลุ่มเพศ
- ไม่ต้องเป็น stereotype หรือ sexualized — โจมตีตรง ๆ ก็เพียงพอ

ต้องผ่านเกณฑ์ A (Gendered Target) + B (Negative Evaluation) และ
ห้ามเป็น meta/counter-speech (C) หรือด่าบุคคลเดียว (D)

กฎเคร่งครัด:
- ส่งออกเฉพาะ JSON ตาม schema
- ข้อความต้องมีตัวระบุเพศ (explicit หรือ implicit ผ่านคำสแลงเพศ)
- ใช้ภาษาโซเชียลมีเดียไทยจริง — ลำลอง reactive หรือ rant
- สร้างความหลากหลายสูงสุด อย่าซ้ำโครงสร้างหรือคำขึ้นต้น
- ข้อความ 1–4 ประโยค สมจริงเหมือนโพสต์/คอมเมนต์จริง
"""

_GB_ATTACK_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยประเภท GB-ATTACK จำนวน {{COUNT}} รายการ

ตัวอย่างเพื่อเป็นแนวทาง (อย่าคัดลอก — ใช้เป็น seed เท่านั้น):
{{SEED_EXAMPLES}}

กระจายเป้าหมายให้หลากหลาย: ผู้หญิง, ผู้ชาย, เกย์, กะเทย, ทอม, ดี้, ไบ, และกลุ่ม LGBTQ+ อื่น ๆ
แต่ละรายการต้องต่างกันอย่างชัดเจนทั้งโครงสร้าง เนื้อหา และเป้าหมาย

บริบทที่ยอมรับ: Twitter rant, คอมเมนต์ข่าว, พันทิป, Facebook, กระทู้ด่า
"""

# ── GB-NORMATIVE ────────────────────────────────────────────────────────────

_GB_NORMATIVE_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียไทยประเภท **GB-NORMATIVE**

คำจำกัดความ GB-NORMATIVE:
ข้อความที่สะท้อนอคติจาก "ค่านิยมทางเพศ" เช่น
- การเหมารวม (stereotype) คุณลักษณะ ความสามารถ นิสัย ตามเพศ
- การกำหนดบทบาททางเพศ (gender role) — ผู้หญิงต้องทำ X, ผู้ชายต้องเป็น Y
- การ policing เพศหรืออัตลักษณ์ — "ทอมที่แท้จริงต้องแมนกว่านี้"
- benevolent stereotype — ยกย่องเชิงแบบแผนที่กำหนดบทบาทตายตัว
ครอบคลุมทั้งเพศหญิง/ชาย และ LGBTQ+

ต้องผ่านเกณฑ์ A + B และห้ามเป็น meta/counter (C) หรือด่าบุคคลเดียว (D)

กฎเคร่งครัด:
- ส่งออกเฉพาะ JSON ตาม schema
- ต้องมีการ generalize เรื่องเพศ (ไม่ใช่บุคคลเดียว)
- อาจสุภาพหรือหยาบก็ได้ แต่ต้องมีนัยกำหนดบทบาท/เหมารวมเพศ
- ใช้ภาษาโซเชียลมีเดียไทยจริง
- ความหลากหลายสูงสุด
"""

_GB_NORMATIVE_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยประเภท GB-NORMATIVE จำนวน {{COUNT}} รายการ

ตัวอย่างเพื่อเป็นแนวทาง (อย่าคัดลอก — ใช้เป็น seed เท่านั้น):
{{SEED_EXAMPLES}}

กระจายประเภทย่อยให้ครบ:
- stereotype (~35%): เหมารวมคุณลักษณะ ความสามารถ อารมณ์ ตามเพศ
- gender_role (~35%): กำหนดสิ่งที่เพศต้องทำ/ห้ามทำ หน้าที่ตามเพศ
- identity_policing (~30%): ตัดสินว่าใครเป็น "เพศจริง" หรือไม่ เช่น "ทอมจริง", "ชายแท้"

กระจายเป้าหมาย: ผู้หญิง, ผู้ชาย, เกย์, กะเทย, ทอม, ดี้ และกลุ่ม LGBTQ+ อื่น ๆ
"""

# ── GB-SEX ──────────────────────────────────────────────────────────────────

_GB_SEX_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียไทยประเภท **GB-SEX**

คำจำกัดความ GB-SEX:
ข้อความที่ลดคุณค่าหรือดูหมิ่นเพศผ่าน:
- sexualized attack — ใช้อวัยวะเพศหรือพฤติกรรมทางเพศเพื่อเหยียด กลุ่มเพศ
- body-based gender insult — วิจารณ์รูปร่าง/ขนาดร่างกายที่ผูกกับเพศในเชิงลบ
- sexual shame / sexual performance shame — ตอกย้ำว่าคุณค่าของเพศขึ้นกับสมรรถภาพทางเพศ
- body-shaming ที่อ้างอิงเพศ
เป้าหมายต้องเป็น กลุ่มเพศ ไม่ใช่บุคคลเดียว (ผ่าน A+B)

ต้องผ่านเกณฑ์ A + B และห้ามเป็น meta/counter (C) หรือด่าบุคคลเดียว (D)
ข้อแตกต่างจาก GB-ATTACK: GB-SEX เน้นที่ร่างกาย อวัยวะเพศ หรือ sexual performance

กฎเคร่งครัด:
- ส่งออกเฉพาะ JSON ตาม schema
- ต้องมีเนื้อหา sexualized ที่ผูกกับกลุ่มเพศ ไม่ใช่บุคคลเดียว
- ใช้ภาษาโซเชียลมีเดียไทยจริง — อาจหยาบคาย
- ความหลากหลายสูงสุด
"""

_GB_SEX_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยประเภท GB-SEX จำนวน {{COUNT}} รายการ

ตัวอย่างเพื่อเป็นแนวทาง (อย่าคัดลอก — ใช้เป็น seed เท่านั้น):
{{SEED_EXAMPLES}}

กระจายประเภทย่อยให้ครบ:
- sexual_insult (~40%): ใช้อวัยวะเพศดูหมิ่นกลุ่มเพศ เช่น จู๋เล็ก, หีดำ, นมปลอม
- body_shame (~35%): วิจารณ์รูปร่างที่ผูกกับเพศในเชิงลบ เช่น กะเทยนมเย้วยาน
- sexual_performance_shame (~25%): ผู้ชายแบบนี้คงไร้น้ำยา, ผู้หญิงพวกนี้ชอบแหกเอง

กระจายเป้าหมาย: ผู้หญิง, ผู้ชาย, กะเทย, เกย์ และกลุ่ม LGBTQ+ อื่น ๆ
"""

# ── NON-GB / Neutral ────────────────────────────────────────────────────────

_NON_GB_NEUTRAL_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียไทยประเภท **NON-GB / Neutral**

คำจำกัดความ:
ข้อความที่ไม่มีเนื้อหาเกี่ยวกับเพศโดยตรงหรือโดยนัยเลย และไม่เข้าเกณฑ์ A/B/C/D
ตัวอย่าง: "วันนี้อากาศดีมาก", "กำลังจะไปทำงาน", "ง่วงนอนสุด ๆ"

ข้อความเหล่านี้ใช้เป็น negative examples ช่วยให้ classifier แยกแยะ GB ออกจาก
ข้อความทั่วไปที่ไม่มีนัยเพศใด ๆ

ให้ครอบคลุมหัวข้อหลากหลาย เช่น:
- การเมือง กีฬา อาหาร จราจร เงิน เทคโนโลยี ดาราบันเทิง งาน (~40%)
- ด่าหรือโจมตีบุคคลเดียว โดยไม่ระบุหรือโยงถึงเพศ (~30%)
- เหมารวมกลุ่มคนโดยอาชีพ อายุ ภูมิภาค หรือชนชั้น — ไม่ใช่เพศ (~30%)

กฎเคร่งครัด:
- ห้ามมีการกล่าวถึงเพศในแง่โจมตี เหมารวม หรือ objectify
- อาจมีคำหยาบหรือความโกรธได้ถ้าไม่เกี่ยวกับเพศ
- ความหลากหลายสูงสุด
"""

_NON_GB_NEUTRAL_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยประเภท NON-GB (Neutral) จำนวน {{COUNT}} รายการ

ตัวอย่างเพื่อเป็นแนวทาง (อย่าคัดลอก — ใช้เป็น seed เท่านั้น):
{{SEED_EXAMPLES}}

กระจายเนื้อหาให้หลากหลาย:
- ข้อความทั่วไปไม่เกี่ยวเพศ: การเมือง กีฬา อาหาร จราจร เงิน โซเชียล การทำงาน เทคโนโลยี
  เช่น "วันนี้รถติดมากเลย", "ราคาข้าวของแพงขึ้นทุกวัน"
- ด่าหรือโจมตีบุคคลเดียว ไม่ระบุเพศ ไม่โยงเพศ
  เช่น "คนอย่างมึงไม่มีอะไรดีหรอก", "แกโง่จริงๆ ทำอะไรก็พัง"
- เหมารวมกลุ่มคนโดยอาชีพ อายุ ภูมิภาค ชนชั้น
  เช่น "นักการเมืองทุกคนมันโกงหมด", "เด็กรุ่นใหม่ไม่รู้จักอดทน"
"""

# ── NON-GB / Meta + Counter-speech ─────────────────────────────────────────

_NON_GB_META_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียไทยประเภท **NON-GB / Meta & Counter-speech**

คำจำกัดความ (เกณฑ์ C ของ annotation-guideline.md):
ข้อความที่มีคำเกี่ยวกับเพศหรืออคติทางเพศ แต่ผู้พูดไม่ได้เหยียดเอง — เป็นการ:
1. meta commentary: กล่าวถึงหรืออ้างข้อความเหยียดของคนอื่น เพื่อวิจารณ์หรือตำหนิ
   เช่น "ยังมีคนพูดว่าผู้หญิงโง่อยู่เลย น่าสมเพชมาก"
2. social critique: วิจารณ์โครงสร้างสังคมหรือค่านิยมที่เหยียดเพศ
   เช่น "สังคมไทยชอบโทษผู้หญิงเวลาโดนคุกคามทางเพศ"
3. counter-speech: โต้แย้งหรือคัดค้านอคติทางเพศ
   เช่น "ใครบอกว่าผู้หญิงขับรถไม่เก่ง นั่นคืออคติล้วน ๆ"

หลักการ: ผู้พูดกำลัง "วิจารณ์/ต่อต้าน" อคติ ไม่ใช่เหยียดเอง → จัดเป็น NON-GB

กฎเคร่งครัด:
- ส่งออกเฉพาะ JSON ตาม schema
- ข้อความต้องชัดเจนว่าผู้พูดไม่ใช่ผู้เหยียด
- อาจมีคำเหยียดในฐานะ "คำพูดที่อ้างถึง" แต่บริบทต้องวิจารณ์หรือโต้แย้ง
- ความหลากหลายสูงสุด
"""

_NON_GB_META_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยประเภท NON-GB (Meta/Counter-speech) จำนวน {{COUNT}} รายการ

ตัวอย่างเพื่อเป็นแนวทาง (อย่าคัดลอก — ใช้เป็น seed เท่านั้น):
{{SEED_EXAMPLES}}

กระจายประเภทให้ครบ:
- meta_commentary (~35%): อ้างคำเหยียดของคนอื่นและวิจารณ์
  เช่น "ยังมีคนโพสว่าเกย์ไม่ควรมีชีวิต น่าสมเพชคนคิดแบบนี้"
- social_critique (~35%): วิจารณ์โครงสร้างหรือค่านิยมสังคมที่เหยียดเพศ
  เช่น "สังคมไทยชอบบอกว่าผู้หญิงไม่ควรทำงานกลางคืน มันคืออคติชัด ๆ"
- counter_speech (~30%): โต้แย้งอคติโดยตรง
  เช่น "ใครบอกว่าผู้ชายห้ามร้องไห้? นั่นแหละคือ toxic masculinity"
"""

# ── NON-GB / Gendered Insult (criterion D) ─────────────────────────────────

_NON_GB_INSULT_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่สร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียไทยประเภท **NON-GB / Gendered Insult**

คำจำกัดความ (เกณฑ์ D ของ annotation-guideline.md):
ข้อความที่มีการกล่าวถึงเพศ + คำดูหมิ่น แต่ยังไม่เป็น GB เพราะ:
- การดูหมิ่นมุ่งไปที่บุคคลหนึ่งคนโดยเฉพาะ ไม่ใช่กลุ่มเพศ
- คำดูหมิ่นไม่ใช่คำเหยียดเพศโดยเนื้อหา (ไม่ใช่ slur เพศ)
- ไม่มีการเหมารวมกลุ่มเพศใด ๆ
- เพศถูกกล่าวถึงเพื่อ "บ่งบอกตัวบุคคล" เท่านั้น ไม่ใช่เป้าหมายของอคติ

ตัวอย่าง:
- "คนนั้นอะ ผู้ชายโง่ๆ" (เป้าหมาย = บุคคลนั้น ไม่ใช่ผู้ชายทั้งหมด)
- "ผู้หญิงคนนั้น ขี้วีนมาก" (บ่นต่อบุคคล ไม่เหมารวมผู้หญิง)
- "เกย์คนนั้น ชอบเถียง" (ไม่ใช่ slur ไม่เหมารวม LGBT)

ห้ามสร้างข้อความที่ใช้ slur เพศ เช่น กะหรี่, ตุ๊ด, ลักเพศ — นั่นเป็น GB-ATTACK

กฎเคร่งครัด:
- ส่งออกเฉพาะ JSON ตาม schema
- ต้องระบุ/บ่งชี้บุคคลเดียวชัดเจน (คนนั้น, เขา, แก, ยายคนนี้ ฯลฯ)
- คำลบต้องไม่ใช่ slur เพศ
- ต้องไม่มีการเหมารวมกลุ่มเพศ
- ความหลากหลายสูงสุด
"""

_NON_GB_INSULT_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยประเภท NON-GB (Gendered Insult) จำนวน {{COUNT}} รายการ

ตัวอย่างเพื่อเป็นแนวทาง (อย่าคัดลอก — ใช้เป็น seed เท่านั้น):
{{SEED_EXAMPLES}}

แต่ละรายการต้อง:
1. มีการระบุเพศ (ผู้ชาย/ผู้หญิง/เกย์/ทอม/ดี้/กะเทย/ฯลฯ) ประกอบบุคคลเดียว
2. มีคำลบ/ดูหมิ่น แต่เจาะจงที่บุคคลนั้น ไม่เหมารวมกลุ่มเพศ
3. คำลบต้องไม่ใช่คำเหยียดเพศโดยเนื้อหา (ห้ามใช้ กะหรี่, ตุ๊ด, ลักเพศ, อีตัว ฯลฯ)
4. ไม่มีนัยเหมารวมว่า "เพศนั้นทุกคนเป็นแบบนี้"

บริบทที่ยอมรับ: บ่นเพื่อน, ด่าเพื่อนบ้าน, คอมเมนต์ถึงคนที่รู้จัก
"""

# ---------------------------------------------------------------------------
# Seed pools — injected via {{SEED_EXAMPLES}} to encourage diversity
# ---------------------------------------------------------------------------

_SEEDS: Dict[str, List[str]] = {
    "gb-attack": [
        "ผู้หญิงมันโง่หมดทุกคน ไม่เชื่อก็ช่าง",
        "ไอ้ตุ๊ดพวกนี้ไม่นับเป็นคนในสายตากู",
        "กะเทยคือของเสียสังคม ควรไปอยู่ที่อื่น",
        "ผู้หญิงพวกนี้ไร้ค่าไร้สมอง มีแต่ปากกับความโง่",
        "ผู้หญิงแบบนี้สมควรโดนทำร้าย ทำตัวยั่วแล้วก็บ่น",
        "ผู้ชายแม่งเหี้ยเหมือนกันหมด ไม่มีตัวไหนดีจริง",
        "เกย์พวกนี้ทำให้สังคมเสื่อมทั้งนั้น",
        "อีพวกลักเพศไม่ควรมีสิทธิ์แบบคนปกติ",
        "ทอมพวกนี้ผิดธรรมชาติ น่ารังเกียจสุดๆ",
        "หญิงพวกนี้ไม่มีคุณค่าอะไรนอกจากใช้งานได้",
        "ผู้ชายที่ร้องไห้คือขยะ ไม่ใช่ผู้ชาย",
        "ไบพวกนี้เห็นแก่ตัว เลือกไม่ได้ ไร้หลักการ",
        "อีตุ๊ดพวกนี้แค่อยากดัง ไม่ได้เป็นจริงสักคน",
        "ผญพวกเฟมทวิตมันหน้าเหี้ยหมดเลย ไม่รู้ทำไม",
        "ดี้พวกนี้สร้างภาพทั้งนั้น ไม่ใช่ของจริง",
    ],
    "gb-normative": [
        "ผู้หญิงดูบอลไม่เป็นหรอก ไม่ต้องมาแสร้งทำ",
        "ผู้ชายต้องเป็นผู้นำเสมอ นั่นคือธรรมชาติ",
        "ทอมที่แท้จริงต้องแมนกว่านี้ อย่ามาทำอ่อนแอ",
        "กะเทยเป็นคนขี้เม้าท์ทุกคน พูดเยอะเกินจริง",
        "ชายแท้ต้องไม่อ่อนแอ ร้องไห้แล้วจะมีใครเคารพ",
        "ผู้หญิงโดยธรรมชาติอ่อนโยน เหมาะกับงานดูแลบ้าน",
        "ผู้ชายต้องปกป้องผู้หญิง นั่นคือหน้าที่ตามเพศ",
        "กระเทยทุกคนต้องน่ารักร่าเริง ดูสนุก",
        "ผู้หญิงไม่ควรทำงานหนัก ปล่อยให้ผู้ชายทำ",
        "เกย์จริงต้องมีนิสัยผู้หญิง ไม่งั้นก็ไม่ใช่เกย์",
        "ผู้ชายที่ดูแลบ้านไม่ใช่ผู้ชายจริงๆ",
        "ผู้หญิงไม่เหมาะกับงานวิศวกร สมองไม่ถึง",
        "ดี้จริงต้องแต่งตัวเท่ ไม่ใช่มาดูอ่อนหวานแบบนี้",
        "ผู้ชายไม่ควรทำอาหาร มันไม่ใช่หน้าที่",
        "เลสที่แท้จริงต้องดุ ไม่ใช่อ่อนหวาน",
    ],
    "gb-sex": [
        "หีดำแบบนี้มันไม่ใช่ผู้หญิงดี ๆ",
        "ควยเล็กแล้วจะเป็นผู้ชายได้ไง",
        "กะเทยนมปลอมเหมือนบอลลูน ดูน่าขยะแขยง",
        "ตุ๊ดพวกนี้ใช้รูตูดอย่างเดียว คุ้มค่าดี",
        "ผู้ชายแบบนี้คงไร้น้ำยา ขนาดนั้น",
        "ก็จู๋เล็กทั้งนั้น เลยต้องทำตัวกร่าง แก้ปม",
        "ผู้หญิงพวกนี้ขายตัวได้ ร่างกายมีอยู่แค่นั้น",
        "กะเทยนมยาน อย่ามาอวดเลย น่าขำ",
        "ผู้หญิงอ้วนพวกนี้ขายตัวไม่ได้หรอก ใครจะเอา",
        "เกย์พวกนี้ตูดหลวมหมดเลย ใช้ซะจนพัง",
        "ผู้ชายไม่มีน้ำยา อยู่ได้ยังไงวะ",
        "หญิงพวกนี้นมเล็ก ไม่น่าสนใจเลย",
        "ตุ๊ดพวกนี้เอาแต่เปิดตูด คุณค่ามีแค่นั้น",
        "ผู้ชายควยกุดพวกนี้ทำอะไรได้บ้าง",
        "ผู้หญิงอ้วนพวกนั้น ถ้าไม่มีร่างกาย มีอะไรเหลือ",
    ],
    "non-gb-neutral": [
        "วันนี้อากาศร้อนมากเลย ออกไปไหนก็เหนื่อย",
        "นักการเมืองทุกคนมันโกงหมด ไม่รู้จะเลือกใคร",
        "คนอย่างมึงไม่มีอะไรดีหรอก ทำอะไรก็ไม่ได้เรื่อง",
        "แกโง่จริงๆ ทำอะไรก็พัง ไม่รู้คิดอะไรอยู่",
        "เด็กรุ่นใหม่ไม่รู้จักอดทนเลย แค่นิดเดียวก็ท้อ",
        "กินข้าวมื้อเย็น ร้านนี้อาหารอร่อยมากเลย",
        "คนรวยพวกนี้ไม่เคยคิดถึงคนจน ใช้เงินสุรุ่ยสุร่าย",
        "รถติดมากเลยวันนี้ ออกมาช้าทีไรก็ซวย",
        "ไอ้คนนี้พูดอะไรก็ไม่เคยฟัง น่ารำคาญสัตว์",
        "เด็กกรุงเทพไม่รู้จักลำบาก ชีวิตสะดวกมากเกินไป",
        "พวกที่ไม่จบปริญญาทำอะไรก็ไม่ได้เรื่อง",
        "ดูซีรีส์จนดึกแล้วตื่นไม่ขึ้น โทษตัวเองไม่ได้",
        "ส่งงานช้ากว่าเส้นตาย เจ้านายจะด่าแน่เลย",
        "ร้านอาหารนี้บริการห่วยมาก ไม่กลับมาอีกแล้ว",
        "มึงคิดอะไรอยู่วะ ทำแบบนี้ไม่ได้เรื่องเลย",
        "พวกอินฟลูเอนเซอร์พวกนี้สร้างภาพหมดเลย",
        "ขอบคุณมากเลยนะ ช่วยได้เยอะมาก",
        "ง่วงนอนสุด ๆ แต่ยังทำงานไม่เสร็จ",
        "กำลังจะไปทำงาน หวังว่าวันนี้จะไม่มีปัญหา",
        "ราคาของแพงขึ้นทุกวัน เงินเดือนไม่พอใช้แล้ว",
    ],
    "non-gb-meta": [
        "ยังมีคนพูดว่าผู้หญิงโง่อยู่เลย น่าสมเพชมาก",
        "สังคมไทยชอบโทษผู้หญิงเวลาโดนคุกคามทางเพศ มันผิดมากเลย",
        "ใครบอกว่าผู้หญิงขับรถไม่เก่ง นั่นคืออคติล้วน ๆ",
        "เห็นคนโพสว่าเกย์ไม่ควรมีสิทธิ์เท่าคนอื่น น่าสมเพชคนคิดแบบนี้",
        "ทำไมสังคมถึงยังมองว่าผู้ชายร้องไห้เป็นเรื่องแปลก",
        "พวกที่บอกว่ากะเทยทำสังคมเสื่อม คิดแบบนี้ได้ยังไง",
        "เหนื่อยใจกับสังคมที่ยังเหยียด LGBTQ+ อยู่",
        "เพราะเหี้ยทั้งหน้าและสันดาน เลยมีปมเรื่องผญ สุดท้ายโดนปฏิเสธ ก็ด่าผญเลย คิคิ",
        "ใครบอกว่าผู้ชายต้องแข็งแกร่งตลอดเวลา มันคือ toxic masculinity",
        "คนที่บอกว่าทอมต้องแมน ไม่เข้าใจเรื่องอัตลักษณ์เลย",
        "อคติเรื่องเพศมันยังฝังรากลึกในสังคมไทยมากเลย",
        "น่าเศร้าที่ยังมีคนมองว่าผู้หญิงไม่ควรเป็นผู้นำ",
        "เห็นคนโพสเหยียดเกย์แล้วรู้สึกหดหู่ สังคมยังไปไม่ถึงไหน",
        "ทำไมถึงยังมีคนคิดว่าผู้หญิงเหมาะแค่อยู่บ้าน ปี 2025 แล้ว",
        "วัฒนธรรมที่บังคับให้ผู้ชายไม่แสดงอารมณ์มันสร้างปัญหามาก",
    ],
    "non-gb-insult": [
        "คนนั้นอะ ผู้ชายโง่ๆ ทำอะไรก็ไม่ได้เรื่อง",
        "ผู้หญิงคนนั้น ขี้วีนมาก อยู่ด้วยกันไม่ได้เลย",
        "ผู้ชายคนนี้แม่ง น่ารำคาญฉิบ พูดอะไรก็ไม่ฟัง",
        "เกย์คนนั้น ชอบเถียง เจอทีไรก็มีเรื่อง",
        "ยายคนนั้น ผู้หญิงเถียงเก่งสัส เหนื่อยมากจริงๆ",
        "ไอ้นี่อะ เกย์งี่เง่า ทำอะไรก็ผิดหมดเลย",
        "แกนะ ผู้หญิงขี้บ่น หยุดบ่นได้แล้ว",
        "ทอมคนนั้นอ่ะ น่ารำคาญจริงๆ ไม่รู้จะว่ายังไง",
        "ผู้ชายคนนั้นหน้ามึน ไม่รู้จักกาลเทศะเลย",
        "ดี้คนนั้นชอบสร้างดราม่า เบื่อมากเลย",
        "เพื่อนผู้หญิงฉันคนนี้ขี้ลืมมาก บอกอะไรก็จำไม่ได้",
        "ไอ้ผู้ชายคนนั้นมันเอาแต่โม้ ฟังแล้วเบื่อ",
        "กะเทยคนนั้นชอบโกหก ไม่น่าไว้ใจเลย",
        "ผู้หญิงคนนั้นขี้เกียจมาก งานไม่เคยเสร็จตรงเวลา",
        "เกย์คนนั้นพูดเยอะจนน่าเบื่อ หยุดได้แล้ว",
    ],
}

# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

_PROMPTS: Dict[str, Dict[str, str]] = {
    "gb-attack":     {"system": _GB_ATTACK_SYSTEM,     "user": _GB_ATTACK_USER},
    "gb-normative":  {"system": _GB_NORMATIVE_SYSTEM,  "user": _GB_NORMATIVE_USER},
    "gb-sex":        {"system": _GB_SEX_SYSTEM,        "user": _GB_SEX_USER},
    "non-gb-neutral":{"system": _NON_GB_NEUTRAL_SYSTEM,"user": _NON_GB_NEUTRAL_USER},
    "non-gb-meta":   {"system": _NON_GB_META_SYSTEM,   "user": _NON_GB_META_USER},
    "non-gb-insult": {"system": _NON_GB_INSULT_SYSTEM, "user": _NON_GB_INSULT_USER},
}

# Expected label values per mode for validation
_EXPECTED_LABEL: Dict[str, str] = {
    "gb-attack":      "GB",
    "gb-normative":   "GB",
    "gb-sex":         "GB",
    "non-gb-neutral": "NON-GB",
    "non-gb-meta":    "NON-GB",
    "non-gb-insult":  "NON-GB",
}

_EXPECTED_SUBTYPE: Dict[str, str] = {
    "gb-attack":      "GB-ATTACK",
    "gb-normative":   "GB-NORMATIVE",
    "gb-sex":         "GB-SEX",
    "non-gb-neutral": "neutral",
    "non-gb-meta":    "meta_counter",
    "non-gb-insult":  "gendered_insult",
}

# ---------------------------------------------------------------------------
# Gemini responseSchema — enforces exact field values at the sampler level
# ---------------------------------------------------------------------------

def _make_response_schema(mode: str) -> Dict[str, Any]:
    """
    Return a JSON-Schema dict suitable for Gemini's generationConfig.responseSchema.
    The outer envelope is always {"items": [...]}.  Each item's shape depends on mode.
    Using enum constraints means the model physically cannot output a wrong label/subtype.
    """
    if mode in ("gb-attack", "gb-normative", "gb-sex"):
        subtype_enum = {
            "gb-attack":    ["GB-ATTACK"],
            "gb-normative": ["GB-NORMATIVE"],
            "gb-sex":       ["GB-SEX"],
        }[mode]
        item_schema = {
            "type": "object",
            "properties": {
                "text":        {"type": "string"},
                "label":       {"type": "string", "enum": ["GB"]},
                "subtype":     {"type": "string", "enum": subtype_enum},
                "bias_target": {"type": "string"},
            },
            "required": ["text", "label", "subtype", "bias_target"],
        }
    else:
        subtype_enum = {
            "non-gb-neutral": ["neutral"],
            "non-gb-meta":    ["meta_counter"],
            "non-gb-insult":  ["gendered_insult"],
        }[mode]
        item_schema = {
            "type": "object",
            "properties": {
                "text":    {"type": "string"},
                "label":   {"type": "string", "enum": ["NON-GB"]},
                "subtype": {"type": "string", "enum": subtype_enum},
            },
            "required": ["text", "label", "subtype"],
        }

    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": item_schema,
            }
        },
        "required": ["items"],
    }

# ---------------------------------------------------------------------------
# JSON parsing / recovery
# ---------------------------------------------------------------------------

def _recover_items(raw: str) -> List[Dict[str, Any]]:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    for key in ('"items"', '"conversations"'):
        idx = cleaned.find(key)
        if idx != -1:
            arr_start = cleaned.find("[", idx)
            if arr_start != -1:
                cleaned = cleaned[arr_start:]
                break

    items: List[Dict[str, Any]] = []
    depth = 0
    in_str = False
    escaped = False
    obj_start: Optional[int] = None

    for i, ch in enumerate(cleaned):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        items.append(json.loads(cleaned[obj_start: i + 1]))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
        elif ch == "]" and depth == 0:
            break
    return items


def _parse_response(raw: str) -> List[Dict[str, Any]]:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        for key in ("items", "conversations", "data", "results"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        return []
    except json.JSONDecodeError:
        return _recover_items(cleaned)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(out_path: Path) -> List[Dict[str, Any]]:
    ckpt = out_path.with_suffix(".ckpt.json")
    for path in (ckpt, out_path):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    print(f"  [resume] loaded {len(data)} items from {path.name}", file=sys.stderr)
                    return data
            except (json.JSONDecodeError, OSError):
                pass
    return []


def _save_checkpoint(items: List[Dict[str, Any]], out_path: Path) -> None:
    ckpt = out_path.with_suffix(".ckpt.json")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ckpt)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def _generate_batch(
    *,
    mode: str,
    count: int,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    seed_sample_size: int = 8,
) -> List[Dict[str, Any]]:
    prompts = _PROMPTS[mode]
    seed_pool = _SEEDS[mode]

    k = min(seed_sample_size, len(seed_pool))
    sampled_seeds = random.sample(seed_pool, k)
    seed_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sampled_seeds))

    system = prompts["system"]
    user = (
        prompts["user"]
        .replace("{{COUNT}}", str(count))
        .replace("{{SEED_EXAMPLES}}", seed_text)
    )

    if dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(system)
        print("\n=== USER PROMPT ===")
        print(user)
        return []

    raw = create_chat_completion(
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
        response_schema=_make_response_schema(mode),
    )
    return _parse_response(raw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_item(item: Any, mode: str) -> bool:
    if not isinstance(item, dict):
        return False
    if not item.get("text", "").strip():
        return False
    expected_label = _EXPECTED_LABEL[mode]
    expected_subtype = _EXPECTED_SUBTYPE[mode]
    if item.get("label") != expected_label:
        return False
    if item.get("subtype") != expected_subtype:
        return False
    return True


# ---------------------------------------------------------------------------
# Main generation loop with checkpoint / resume + parallel waves
# ---------------------------------------------------------------------------

def _generate_batch_with_retry(
    *,
    batch_num: int,
    mode: str,
    count: int,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    retry_delay: float = 5.0,
    max_attempts: int = 5,
) -> List[Dict[str, Any]]:
    for attempt in range(max_attempts):
        try:
            return _generate_batch(
                mode=mode,
                count=count,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                dry_run=dry_run,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                wait = retry_delay * (attempt + 1) * 2
                print(
                    f"  [batch {batch_num}] rate-limited, waiting {wait:.0f}s …",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            if attempt == max_attempts - 1:
                raise
            print(
                f"  [batch {batch_num}] error (attempt {attempt+1}/{max_attempts}): {exc}",
                file=sys.stderr,
            )
            time.sleep(retry_delay * (attempt + 1))
    return []


def generate_all(
    *,
    mode: str,
    total: int,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    out_path: Path,
    parallel: int = 1,
    retry_delay: float = 5.0,
) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = _load_checkpoint(out_path) if not dry_run else []
    seen_texts = {i.get("text", "") for i in all_items}
    remaining = total - len(all_items)
    batch_num = 0

    if remaining <= 0:
        print(f"  [resume] already have {len(all_items)} items, nothing to do.", file=sys.stderr)
        return all_items[:total]

    if all_items:
        print(f"  [resume] continuing from {len(all_items)}/{total} …", file=sys.stderr)

    while remaining > 0:
        # How many parallel calls to fire this wave
        calls_this_wave = min(parallel, -(-remaining // BATCH_SIZE))  # ceil div

        # Distribute items across calls
        pending = remaining
        counts: List[int] = []
        for _ in range(calls_this_wave):
            c = min(BATCH_SIZE, pending)
            counts.append(c)
            pending -= c

        wave_batches: List[tuple[int, int]] = []
        for c in counts:
            batch_num += 1
            wave_batches.append((batch_num, c))
            print(
                f"[batch {batch_num}] requesting {c} items "
                f"(collected {len(all_items)}/{total})",
                file=sys.stderr,
            )

        if dry_run:
            _generate_batch(
                mode=mode, count=counts[0], api_key=api_key, model=model,
                temperature=temperature, max_tokens=max_tokens, dry_run=True,
            )
            break

        wave_items: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=calls_this_wave) as executor:
            futures = [
                executor.submit(
                    _generate_batch_with_retry,
                    batch_num=bnum,
                    mode=mode,
                    count=bcount,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    dry_run=False,
                    retry_delay=retry_delay,
                )
                for bnum, bcount in wave_batches
            ]
            for future in as_completed(futures):
                wave_items.extend(future.result())

        # Deduplicate + validate
        fresh = [i for i in wave_items if i.get("text", "") not in seen_texts]
        valid = [i for i in fresh if _validate_item(i, mode)]

        if not valid:
            print(
                f"  warning: no valid items in wave (got {len(wave_items)}, "
                f"fresh {len(fresh)}), retrying …",
                file=sys.stderr,
            )
            time.sleep(retry_delay)
            continue

        all_items.extend(valid)
        seen_texts.update(i.get("text", "") for i in valid)
        remaining = total - len(all_items)

        _save_checkpoint(all_items, out_path)
        print(f"  [checkpoint] {len(all_items)}/{total} saved", file=sys.stderr)

        if remaining > 0:
            time.sleep(1.0)

    return all_items[:total]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Load .env from cwd first, then fall back to package-local .env
    _load_env(Path.cwd() / ".env")
    _load_env(DEFAULT_ENV_FILE)

    parser = argparse.ArgumentParser(
        description="Synthesize Thai gender-bias data aligned with annotation-guideline.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=list(_PROMPTS.keys()),
        help=(
            "Generation mode: "
            "gb-attack | gb-normative | gb-sex | "
            "non-gb-neutral | non-gb-meta | non-gb-insult"
        ),
    )
    parser.add_argument("--count", type=int, default=100, help="Total samples to generate. Default: 100")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file. Defaults to output/<mode>.json inside the package.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID. Default: {DEFAULT_MODEL}")

    # ── Provider selection ──────────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "gemini-batch"],
        help=(
            "Inference provider. "
            "'openrouter' (default): live OpenRouter API calls. "
            "'gemini-batch': Vertex AI async batch prediction (50%% cost discount)."
        ),
    )

    # ── OpenRouter / live options ───────────────────────────────────────────
    parser.add_argument("--api-key", default=None, help="Gemini API key (falls back to GEMINI_API_KEY env var)")

    # ── Gemini Batch / Vertex AI options ───────────────────────────────────
    parser.add_argument(
        "--gcp-project",
        default=None,
        help="[gemini-batch] GCP project ID (falls back to GCP_PROJECT env var)",
    )
    parser.add_argument(
        "--gcp-location",
        default="us-central1",
        help="[gemini-batch] Vertex AI region. Default: us-central1",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=None,
        help="[gemini-batch] GCS bucket name for input/output JSONL (no gs:// prefix)",
    )
    parser.add_argument(
        "--gcs-prefix",
        default="synthesizer-v2",
        help="[gemini-batch] GCS path prefix inside the bucket. Default: synthesizer-v2",
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=int,
        default=60,
        help="[gemini-batch] Seconds between job state polls. Default: 60",
    )

    # ── Common options ──────────────────────────────────────────────────────
    parser.add_argument("--temperature", type=float, default=0.95, help="Sampling temperature. Default: 0.95")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens per call. Default: 8192")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel API calls per wave. Default: 3")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, no API call")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Gemini Batch provider ───────────────────────────────────────────────
    if args.provider == "gemini-batch":
        gcp_project: str = args.gcp_project or os.environ.get("GCP_PROJECT", "")
        if not gcp_project and not args.dry_run:
            print(
                "Error: --gcp-project (or GCP_PROJECT env var) is required for gemini-batch.",
                file=sys.stderr,
            )
            sys.exit(1)

        gcs_bucket: str = args.gcs_bucket or os.environ.get("GCS_BUCKET", "")
        if not gcs_bucket and not args.dry_run:
            print(
                "Error: --gcs-bucket (or GCS_BUCKET env var) is required for gemini-batch.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"Mode: {args.mode} | Provider: gemini-batch | Model: {args.model} "
            f"| Count: {args.count} | Output: {out_path}",
            file=sys.stderr,
        )

        from synthesizer_v2.gemini_batch import run_batch_pipeline

        items = run_batch_pipeline(
            mode=args.mode,
            total=args.count,
            model=args.model,
            project=gcp_project,
            location=args.gcp_location,
            gcs_bucket=gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            prompts=_PROMPTS,
            seeds=_SEEDS,
            expected_label=_EXPECTED_LABEL[args.mode],
            expected_subtype=_EXPECTED_SUBTYPE[args.mode],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            poll_interval=args.batch_poll_interval,
            out_path=out_path,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print("\n[dry-run] No output written.", file=sys.stderr)
            return

    # ── Gemini live provider ────────────────────────────────────────────────
    else:
        api_key: str = args.api_key or os.environ.get("GEMINI_API_KEY", "")

        if not api_key and not args.dry_run:
            print(
                "Error: no API key found. Set GEMINI_API_KEY in .env or pass --api-key.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"Mode: {args.mode} | Model: {args.model} | Count: {args.count} "
            f"| Parallel: {args.parallel} | Output: {out_path}",
            file=sys.stderr,
        )

        items = generate_all(
            mode=args.mode,
            total=args.count,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            out_path=out_path,
            parallel=args.parallel,
        )

        if args.dry_run:
            print("\n[dry-run] No output written.", file=sys.stderr)
            return

    # ── Write output ────────────────────────────────────────────────────────
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    # Remove checkpoint on success
    ckpt = out_path.with_suffix(".ckpt.json")
    if ckpt.exists():
        ckpt.unlink()

    # Summary
    subtype_counts: Dict[str, int] = {}
    for item in items:
        sub = item.get("subtype", "unknown")
        subtype_counts[sub] = subtype_counts.get(sub, 0) + 1

    print(f"\nSaved {len(items)} items → {out_path}", file=sys.stderr)
    print(f"Subtype distribution: {subtype_counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
