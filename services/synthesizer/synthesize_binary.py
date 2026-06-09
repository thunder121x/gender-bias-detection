"""
synthesize_binary.py
====================
Generate synthetic Thai social-media data with three labels:
  label=1  → gender-biased          (use --bias)
  label=0  → non-gender-biased      (use --neutral)
  label=2  → gender-related but not biased  (use --gender-discuss)

Bias subtypes covered
---------------------
  sexism        – rude, hostile attacks on gender; may use Thai vulgar/slang
                  terms whose meaning objectifies gender (e.g. กะหรี่, ตุ๊ด used
                  as slurs, กระสัน-type insults tied to gender)
  gender_object – reducing a person to a sexual/body object purely because of
                  gender; crude body-based insults are expected/encouraged here
  stereotypes   – assigning abilities, roles, or norms to a gender group;
                  can be rude OR casual/polite; the key is the gender-norm
                  generalisation (ผู้หญิงทำอาหารเก่ง, ผู้ชายไม่ร้องไห้, etc.)

Neutral mode
------------
  --neutral generates posts that may contain Thai profanity, anger, insults,
  crude language, or controversial opinions – but WITHOUT targeting gender.
  The offence must be about something else (personality, politics, sport,
  food, traffic, money, …).

Gender-discuss mode
-------------------
  --gender-discuss generates posts that mention gender but are NOT biased:
  counter-speech pushing back on gender bias, open questions about gender
  norms, awareness/educational posts, and personal self-identification.

Usage
-----
  # 100 bias samples via Gemini through OpenRouter → output/gemini/bias_samples.json
  python synthesize_binary.py --bias --provider openrouter --model google/gemini-2.5-flash-001 --output output/gemini/bias_samples.json

  # 100 neutral samples
  python synthesize_binary.py --neutral --provider openrouter --model google/gemini-2.5-flash-001 --output output/gemini/neutral_samples.json

  # 100 gender-discuss samples
  python synthesize_binary.py --gender-discuss --provider openrouter --model google/gemini-2.5-flash-001 --output output/gemini/gender_discuss_samples.json

  # Use direct Gemini API (needs GEMINI_API_KEY)
  python synthesize_binary.py --bias --provider gemini

  # Use OpenAI GPT API (needs OPENAI_API_KEY)
  python synthesize_binary.py --bias --provider gpt

  # Use OpenRouter (needs OPENROUTER_API_KEY) – supports any model on openrouter.ai
  python synthesize_binary.py --bias --provider openrouter --model google/gemini-2.5-flash-001

  # Dry-run (print prompt, no API call)
  python synthesize_binary.py --bias --dry-run

Environment variables (or .env file in project root)
-----------------------------------------------------
  GEMINI_API_KEY      – Google AI Studio / Vertex AI key
  OPENAI_API_KEY      – OpenAI key
  OPENROUTER_API_KEY  – OpenRouter key (supports Gemini, GPT, Claude, etc.)
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
from urllib import error, request

# ---------------------------------------------------------------------------
# .env loader (no external deps)
# ---------------------------------------------------------------------------


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)


_load_env(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BIAS_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่กำลังสร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานวิจัยการตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียภาษาไทยที่มีอคติทางเพศอย่างชัดเจน

ประเภทของอคติที่ต้องครอบคลุม (สลับกันในแต่ละ batch):
1. sexism – ข้อความเกลียดชัง ดูหมิ่น หรือโจมตีเพศโดยตรง อาจใช้คำแสลงหยาบคาย
   ที่มีความหมายเหยียดเพศตามบริบทไทย เช่น กะหรี่ ตุ๊ด หน้าหีหรือคำที่ผู้ใช้งาน
   โซเชียลมีเดียใช้จริงในการโจมตีเพศ
2. gender_object – ลดคุณค่าบุคคลให้เป็นวัตถุทางเพศ/ร่างกาย เนื่องจากเพศของเขา
   ข้อความหยาบคายและรุนแรงได้ เน้นที่การ objectify ผ่านเพศ
3. stereotypes – กำหนดบทบาท ความสามารถ หรือนิสัยให้กับกลุ่มเพศ จะหยาบคายหรือ
   ไม่ก็ได้ แต่ต้องเป็นการเหมารวมเรื่องเพศ เช่น ผู้หญิงทำอาหารเก่ง ผู้ชายไม่ร้องไห้
   เกย์ไม่แมน ผู้หญิงขับรถแย่

กฎที่ต้องปฏิบัติอย่างเคร่งครัด:
- ส่งออกเฉพาะ JSON ที่ถูกต้องตาม schema
- อคติต้องชัดเจน มีตัวระบุเพศ (ผู้หญิง ผู้ชาย เกย์ กะเทย ทอม ดี้ ฯลฯ)
- ใช้ภาษาโซเชียลมีเดียไทยจริง: ลำลอง อารมณ์ reactive หรือ rant ตามบริบท
- ห้ามสร้างข้อความเป็นกลาง ห้ามอธิบายอคติ ห้ามยกย่อง
- สร้างความหลากหลายสูงสุด: อย่าทำซ้ำโครงสร้างประโยค
- ข้อความแต่ละชิ้นควรเป็นโพสต์/คอมเมนต์โซเชียลมีเดียสมจริง (ความยาว 1-4 ประโยค)
"""

BIAS_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยที่มีอคติทางเพศจำนวน {{COUNT}} รายการ

กระจายประเภทให้ครบ: sexism, gender_object, stereotypes
แต่ละรายการต้องต่างกันอย่างชัดเจนทั้งโครงสร้างและเนื้อหา

กฎสำหรับ sexism:
- ใช้คำโจมตีเพศโดยตรง อาจรวมถึงคำแสลงหยาบคาย
- ต้องมีความเกลียดชังหรือดูหมิ่นที่ผูกกับเพศ
- ตัวอย่างบริบท: Twitter rant, คอมเมนต์ข่าว, พันทิป

กฎสำหรับ gender_object:
- ลดคุณค่าบุคคลให้เป็นวัตถุทางเพศเพราะเพศของเขา
- อาจใช้คำหยาบที่อ้างอิงร่างกายผ่านเลนส์เพศ
- บริบท: คอมเมนต์ใต้รูป, แชท, ห้องคุย

กฎสำหรับ stereotypes:
- เหมารวมความสามารถ บทบาท หรือนิสัยผ่านเพศ
- อาจสุภาพหรือหยาบก็ได้ แต่ต้องมีการ generalize เรื่องเพศ
- บริบท: คอมเมนต์ เถียงกัน แชร์ความเห็น

รูปแบบ JSON ที่ต้องการ:
{
  "items": [
    {
      "text": "ข้อความโซเชียลมีเดียภาษาไทย",
      "label": 1,
      "subtype": "sexism | gender_object | stereotypes",
      "bias_target": "gender group targeted (เช่น ผู้หญิง, ผู้ชาย, เกย์, กะเทย)"
    }
  ]
}

ส่งออกเฉพาะ JSON เท่านั้น ห้ามอธิบาย ห้ามใส่ markdown
จำนวน items ต้องเท่ากับ {{COUNT}} พอดี
"""

NEUTRAL_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่กำลังสร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานวิจัยการตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียภาษาไทยที่ไม่มีอคติทางเพศ

ข้อความเหล่านี้ใช้เป็น negative examples สำหรับ classifier
จุดสำคัญ: อาจมีคำหยาบ ความโกรธ การวิจารณ์ ความรุนแรงทางวาจา หรือเนื้อหาอื่นๆ ที่ไม่เกี่ยวข้องกับเพศ
แต่ต้องไม่มีการโจมตี เหมารวม หรือ objectify บุคคลเนื่องจากเพศของพวกเขา

ประเภทของ neutral ที่ต้องครอบคลุม:
1. topic_based – ข้อความเกี่ยวกับหัวข้อที่ไม่ใช่เพศ เช่น การเมือง กีฬา อาหาร จราจร เงิน
   โซเชียลมีเดีย influencer การทำงาน เจ้านาย เทคโนโลยี แอป สัตว์เลี้ยง เพื่อนบ้าน
   ดาราบันเทิง (โดยไม่เชื่อมกับเพศ) — อาจสุภาพหรือหยาบคายก็ได้
2. personal_insult – ด่าหรือโจมตีบุคคลหนึ่งโดยเฉพาะ ไม่ใช่กลุ่มเพศ เน้นนิสัย พฤติกรรม
   ความโง่ ความขี้เกียจ ความน่ารำคาญ ไม่มีการระบุหรือเชื่อมกับเพศเลย
3. non_gender_stereotype – เหมารวมกลุ่มคนโดยอาศัยปัจจัยอื่นที่ไม่ใช่เพศ เช่น อาชีพ รายได้
   อายุ ภูมิภาค การศึกษา รสนิยม พฤติกรรมการใช้เงิน
4. short_reaction – คอมเมนต์สั้นๆ ที่เป็นเพียงปฏิกิริยา ความรู้สึก หรือการขอบคุณทั่วไป
   ไม่มีเนื้อหาโจมตี ไม่มีการกล่าวถึงเพศ เช่น "ขอบคุณมากค่ะ", "ว้าวมากเลย",
   "ชัดเจนมากเลยค่ะ", "โอเคมากเลย", "พูดดีจริงๆ", "เห็นด้วยเลย", "สรุปดีมากค่ะ"
5. relationship_advice – คำแนะนำหรือความเห็นเรื่องความสัมพันธ์ที่ไม่ได้โจมตีหรือเหมารวมเพศ
   อาจพูดถึงคนหรือพฤติกรรมเฉพาะโดยไม่กล่าวโทษเพศ เช่น "รักตัวเองก่อนนะ",
   "ถ้าเขาไม่ดีก็ตัดทิ้งเลย ไม่ต้องเสียเวลา", "คนที่ดีมีอยู่ แค่ต้องเลือกให้ถูก",
   "อยู่คนเดียวก็มีความสุขได้นะ ไม่ต้องรีบมีแฟน"

กฎที่ต้องปฏิบัติ:
- ส่งออกเฉพาะ JSON ที่ถูกต้องตาม schema
- ห้ามกล่าวถึงเพศในแง่ของการโจมตี เหมารวม หรือ objectify
- อาจมีคำหยาบหรือความโกรธได้ถ้าไม่ใช่เรื่องเพศ
- ความหลากหลายสูงสุด: อย่าทำซ้ำหัวข้อหรือโครงสร้าง
- ใช้ภาษาโซเชียลมีเดียไทยจริง
"""

# Pool of seed examples drawn randomly per batch to force variance
_NEUTRAL_SEED_POOL: List[str] = [
    "ขอบคุณมากค่ะ ได้ความรู้เยอะเลย",
    "ว้าวมากเลย ไม่เคยคิดมุมนี้มาก่อน",
    "ชัดเจนมากค่ะ เข้าใจเลย",
    "โอเคมากเลยนะ ลองทำดูแล้วกัน",
    "สรุปดีมากเลยค่ะ",
    "พูดดีมาก ใช่เลย",
    "เห็นด้วยทุกข้อเลย",
    "รักตัวเองก่อนนะคะ สำคัญที่สุด",
    "ถ้าเขาไม่ดีก็ตัดทิ้งเลย ไม่ต้องเสียเวลา",
    "คนที่ดีมีเยอะ แค่ต้องรอให้ถูกคน",
    "อยู่คนเดียวก็มีความสุขได้นะ ไม่ต้องรีบ",
    "แนะนำได้ดีมากเลย จะลองทำตามดู",
    "เสียเวลาแน่นอนถ้าคบกับคนไม่ดี ออกมาเลยดีกว่า",
    "ใช่เลยค่ะ รักตัวเองให้มากๆ นะ",
    "เพิ่งมาเจอช่องนี้ ดีมากเลยค่ะ",
    "ขอบคุณนะคะ ได้แนวคิดใหม่เลย",
    "จริงๆ เลยนะ อยู่คนเดียวสบายกว่าเยอะ",
    "ไม่มีแฟนก็ไม่เป็นไร ชีวิตยังดำเนินต่อได้",
    "ถนนติดมากเลยวันนี้짜증จริงๆ",
    "ค่าครองชีพขึ้นอีกแล้ว ไม่ไหวแล้ว",
    "ร้านนี้บริการแย่มากเลย ไม่มาอีกแล้ว",
    "เน็ตช้ามากเลยวันนี้ ใช้งานไม่ได้เลย",
    "เจ้านายบ้าจริงๆ สั่งงานตอนดึกอีก",
    "แอปนี้ crash ตลอดเลย แย่มาก",
    "นักการเมืองพูดแต่ไม่ทำ เบื่อจริงๆ",
    "ทีมแพ้อีกแล้ว ผิดหวังมาก",
    "ราคาข้าวแพงขึ้นทุกวัน กินอะไรดีวะ",
    "คนโง่ไม่ต้องไปสนนะ เสียเวลา",
    "ไอ้นี่มันทำงานไม่ได้เรื่องจริงๆ ไม่รู้จะอธิบายยังไง",
    "อีพวกนี้ผลาญเงินพ่อแม่เล่น ไม่รู้จักรับผิดชอบ",
]

NEUTRAL_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยที่ไม่มีอคติทางเพศจำนวน {{COUNT}} รายการ

ข้อความเหล่านี้คือ non-bias examples สำหรับชุดข้อมูลการวิจัย
อาจมีคำหยาบ ความโกรธ หรือความคิดเห็นแรงๆ ได้ แต่ต้องไม่โจมตีหรือเหมารวมเพศ

กระจายประเภทให้ครบทั้งห้าแบบ:

ประเภท topic_based (~25%):
- เนื้อหาเกี่ยวกับการเมือง กีฬา อาหาร จราจร เงิน โซเชียลมีเดีย การทำงาน เทคโนโลยี ฯลฯ
- อาจสุภาพหรือหยาบคายก็ได้ แต่ไม่เกี่ยวกับเพศ

ประเภท personal_insult (~20%):
- ด่าหรือโจมตีบุคคลหนึ่ง ไม่ใช่กลุ่มเพศ เน้นนิสัย พฤติกรรม ความโง่ ความน่ารำคาญ
- ตัวอย่าง: "คนอย่างมึงไม่มีอะไรดีหรอก", "แกโง่จริงๆ ทำอะไรก็พัง",
  "ไอ้คนนี้พูดอะไรก็ไม่เคยฟัง น่ารำคาญสัตว์"

ประเภท non_gender_stereotype (~20%):
- เหมารวมกลุ่มคนโดยอาศัย อาชีพ รายได้ อายุ ภูมิภาค การศึกษา รสนิยม พฤติกรรม — ไม่ใช่เพศ
- ตัวอย่าง: "อีพวกนี้ ผลาญเงินพ่อแม่เล่น", "เด็กรุ่นใหม่ไม่รู้จักอดทนเลย",
  "คนรวยพวกนี้ไม่เคยคิดถึงคนจน", "นักการเมืองทุกคนมันโกงหมด"

ประเภท short_reaction (~15%):
- คอมเมนต์สั้นๆ ที่เป็นเพียงปฏิกิริยา ความรู้สึก หรือการขอบคุณทั่วไป
- ไม่มีเนื้อหาโจมตี ไม่มีการกล่าวถึงเพศเลย
- ตัวอย่าง: "ขอบคุณมากค่ะ", "ว้าวมากเลย", "ชัดเจนดีมากเลยค่ะ", "เห็นด้วยเลย",
  "พูดดีมากจริงๆ", "โอเคมากเลยนะ", "สรุปดีมากค่ะ", "ได้ความรู้มากเลย"

ประเภท relationship_advice (~20%):
- คำแนะนำหรือความเห็นเรื่องความสัมพันธ์ที่ไม่โจมตีหรือเหมารวมเพศ
- อาจพูดถึงพฤติกรรมของ "คน" หรือ "เขา/เธอ" แต่ไม่กล่าวโทษเพศนั้นๆ โดยรวม
- ตัวอย่าง: "รักตัวเองก่อนนะ อย่าไปง้อคนที่ไม่ได้รัก",
  "ถ้าคบกับคนไม่ดีก็ตัดทิ้งเลย ไม่ต้องเสียเวลา",
  "อยู่คนเดียวก็มีความสุขได้ ไม่ต้องรีบมีแฟน",
  "คนที่ดีมีเยอะ แค่ต้องเลือกให้ถูก",
  "เลิกกับคนนอกใจได้เลย ไม่ต้องอธิบาย"

ตัวอย่างเพิ่มเติมที่ต้องการความหลากหลายแบบนี้:
{{SEED_EXAMPLES}}

รูปแบบ JSON ที่ต้องการ:
{
  "items": [
    {
      "text": "ข้อความโซเชียลมีเดียภาษาไทย",
      "label": 0,
      "topic": "ประเภทและหัวข้อ (เช่น topic_based/การเมือง, personal_insult, non_gender_stereotype/อายุ, short_reaction, relationship_advice)"
    }
  ]
}

ส่งออกเฉพาะ JSON เท่านั้น ห้ามอธิบาย ห้ามใส่ markdown
จำนวน items ต้องเท่ากับ {{COUNT}} พอดี
"""

_GENDER_DISCUSS_SEED_POOL: List[str] = [
    # counter_speech
    "ที่บอกว่าผู้หญิงขับรถแย่มันไม่จริงเลย สถิติก็ไม่ได้บอกแบบนั้น",
    "เกย์ก็เป็นคนเหมือนกัน ไม่เข้าใจว่าทำไมต้องมองต่าง",
    "ผู้ชายร้องไห้ได้นะ ไม่ได้แปลว่าอ่อนแอ",
    "การที่สังคมคาดหวังให้ผู้ชายไม่ร้องไห้มันกดดันมากเลย",
    "ทำไมต้องมองว่าผู้หญิงทำงานสาย STEM ได้น้อยกว่าผู้ชาย ตอนนี้มีเยอะมากแล้วนะ",
    "พวกที่บอกว่ากะเทยเป็นแค่แฟชั่น ไม่เข้าใจอะไรเลย อัตลักษณ์มันไม่ใช่เทรนด์",
    "ผู้หญิงเป็นผู้นำได้ดีพอกันหรือดีกว่าก็มี ไม่ต้องตั้งคำถามเรื่องเพศ",
    # gender_question
    "ทำไมผู้ชายถึงต้องเป็นคนจ่ายตลอดเลยนะ ยุคนี้แล้ว",
    "รู้สึกว่าสังคมไทยยังกดดันเรื่องบทบาทเพศเยอะมาก ใครรู้สึกแบบนี้บ้าง",
    "ทำไมอาชีพพยาบาลถึงยังถูกมองว่าเป็นของผู้หญิงอยู่เลย",
    "สงสัยว่าทำไมสังคมยังแปลกใจเวลาเห็นพ่อบ้านอยู่กับลูกแทนแม่",
    "ถ้าผู้หญิงก้าวร้าวจะโดนตัดสิน แต่ถ้าผู้ชายทำแบบเดียวกันกลายเป็น assertive ทำไมนะ",
    # awareness
    "ความรุนแรงในครอบครัวเกิดกับทุกเพศได้นะคะ อย่ามองข้ามผู้ชายที่โดนด้วย",
    "LGBT+ ก็มีสิทธิ์มีความสุขเหมือนกัน ไม่ใช่เรื่องผิดปกติ",
    "เพศไม่ควรกำหนดว่าเราจะทำอาชีพอะไรได้ ขอให้ทำได้ดีก็พอ",
    "การ harass คนในที่ทำงานเพราะเพศมันผิดกฎหมายนะ รู้หรือเปล่า",
    "วันนี้เป็น International Women's Day ขอส่งกำลังใจให้ทุกคนที่ยังต้องสู้เพื่อความเท่าเทียมนะ",
    "non-binary ไม่ใช่เรื่องใหม่ มีมาในหลายวัฒนธรรมทั่วโลกนานมากแล้ว",
    # self_identification
    "เพิ่งคุยกับพ่อแม่เรื่องที่เราเป็น bi ได้แล้ว โล่งใจมากเลย",
    "ไม่ได้รู้สึกตัวเองเป็นผู้หญิง 100% มาตลอด แต่ก็โอเคกับตัวเองนะ",
    "ใช้เวลานานมากกว่าจะยอมรับตัวเองได้ว่าเป็น trans แต่ตอนนี้ดีขึ้นมากแล้ว",
    "เพิ่งรู้ว่าตัวเองเป็น asexual ก็รู้สึกว่าหลายอย่างในชีวิตมันสมเหตุสมผลขึ้นเลย",
    "เราเป็น gay มาตลอด แค่ยังไม่พร้อม come out กับทุกคน ขอเวลาก่อนนะ",
    "ตัวเองเป็น non-binary มานานแล้ว แค่ไม่รู้คำนี้จนได้เจอคอมมูนิตี้",
    "เพิ่ง accept ตัวเองเรื่อง gender identity ได้เมื่อปีที่แล้ว รู้สึกเบาใจมาก",
]

GENDER_DISCUSS_SYSTEM = """\
คุณคือผู้ช่วยวิจัย NLP ที่กำลังสร้างข้อมูลสังเคราะห์ภาษาไทยสำหรับงานวิจัยการตรวจจับอคติทางเพศ
งานของคุณคือสร้างข้อความโซเชียลมีเดียภาษาไทยที่เกี่ยวข้องกับเพศ แต่ไม่มีอคติทางเพศ

ข้อความเหล่านี้คือ label=2 (gender-related, non-biased) สำหรับ classifier
ข้อความต้องกล่าวถึงเพศหรืออัตลักษณ์ทางเพศ แต่ต้องเป็นในเชิงปกป้อง ตั้งคำถาม ให้ความรู้ หรือเล่าประสบการณ์ส่วนตัว
ห้ามโจมตี ดูหมิ่น หรือเหมารวมเพศใดเพศหนึ่ง

ประเภทของ gender-discuss ที่ต้องครอบคลุม:
1. counter_speech – โต้แย้งหรือปกป้องต่ออคติทางเพศที่มีอยู่ในสังคม
   เช่น "ที่บอกว่าผู้หญิงขับรถแย่มันไม่จริงเลย สถิติก็ไม่ได้บอกแบบนั้น"
   "เกย์ก็เป็นคนเหมือนกัน ไม่เข้าใจว่าทำไมต้องมองต่าง"
   "การที่สังคมคาดหวังให้ผู้ชายไม่ร้องไห้มันกดดันมากเลยนะ"
2. gender_question – ตั้งคำถามหรือถกเถียงเรื่องบรรทัดฐานทางเพศอย่างเปิดกว้าง
   ไม่ได้โจมตีใคร แค่ตั้งประเด็นหรืองงว่าทำไมสังคมถึงเป็นแบบนั้น
   เช่น "ทำไมผู้ชายถึงต้องเป็นคนจ่ายตลอดเลยนะ ยุคนี้แล้ว"
   "รู้สึกว่าสังคมไทยยังกดดันเรื่องบทบาทเพศเยอะมาก ใครรู้สึกแบบนี้บ้าง"
3. awareness – โพสต์ให้ความรู้ ให้ข้อมูล หรือรณรงค์เรื่องความเท่าเทียมทางเพศ
   เช่น "ความรุนแรงในครอบครัวเกิดกับทุกเพศได้ อย่ามองข้ามนะคะ"
   "LGBT+ ก็มีสิทธิ์มีความสุขเหมือนกัน ไม่ใช่เรื่องผิดปกติ"
4. self_identification – เล่าหรือแสดงออกถึงอัตลักษณ์ทางเพศของตัวเองโดยไม่โจมตีผู้อื่น
   เช่น "เพิ่งคุยกับพ่อแม่เรื่องที่เราเป็น bi ได้แล้ว โล่งใจมากเลย"
   "ไม่ได้รู้สึกตัวเองเป็นผู้หญิง 100% มาตลอด แต่ก็โอเคกับตัวเองนะ"

กฎที่ต้องปฏิบัติอย่างเคร่งครัด:
- ส่งออกเฉพาะ JSON ที่ถูกต้องตาม schema
- ข้อความต้องกล่าวถึงเพศหรืออัตลักษณ์ทางเพศอย่างชัดเจน
- ห้ามโจมตี ดูหมิ่น หรือ objectify เพศใด
- ห้ามเหมารวมในทางลบ (stereotype ที่เป็นอคติ)
- ใช้ภาษาโซเชียลมีเดียไทยจริง: อาจลำลอง สะท้อน หรือแสดงอารมณ์ได้
- สร้างความหลากหลายสูงสุด: อย่าทำซ้ำโครงสร้างหรือเนื้อหา
"""

GENDER_DISCUSS_USER = """\
สร้างข้อความโซเชียลมีเดียภาษาไทยที่เกี่ยวกับเพศแต่ไม่มีอคติทางเพศจำนวน {{COUNT}} รายการ

ข้อความเหล่านี้คือ label=2 (gender-related, non-biased) ต้องมีคำหรือบริบทเกี่ยวกับเพศ
แต่ต้องเป็นเชิงปกป้อง ตั้งคำถาม ให้ความรู้ หรือเล่าประสบการณ์ — ไม่ใช่โจมตีหรือเหมารวม

กระจายประเภทให้ครบทั้งสี่แบบ:

ประเภท counter_speech (~25%):
- โต้แย้งหรือปฏิเสธอคติทางเพศที่พบในสังคม
- ตัวอย่าง: "ที่บอกว่าผู้หญิงขับรถแย่มันไม่จริงเลย",
  "เกย์ก็เป็นคนเหมือนกัน ทำไมต้องมองต่าง",
  "ผู้ชายร้องไห้ได้ ไม่ได้แปลว่าอ่อนแอ"

ประเภท gender_question (~25%):
- ตั้งคำถามหรือสะท้อนบรรทัดฐานทางเพศอย่างเปิดกว้าง ไม่โจมตีใคร
- ตัวอย่าง: "ทำไมผู้ชายต้องเป็นคนจ่ายตลอดเลยนะ",
  "รู้สึกว่าสังคมยังกดดันเรื่องบทบาทเพศอยู่มากเลย ใครรู้สึกแบบนี้บ้าง",
  "ทำไมอาชีพบางอย่างถึงยังถูกมองว่าเป็นของผู้ชายอยู่"

ประเภท awareness (~25%):
- ให้ข้อมูล รณรงค์ หรือแชร์มุมมองเชิงบวกเรื่องความเท่าเทียมทางเพศ
- ตัวอย่าง: "ความรุนแรงในครอบครัวเกิดกับทุกเพศได้นะคะ อย่ามองข้าม",
  "LGBT+ ก็มีสิทธิ์มีความสุข ไม่ใช่เรื่องผิดปกติ",
  "เพศไม่ควรกำหนดว่าเราจะทำอาชีพอะไรได้"

ประเภท self_identification (~25%):
- เล่าหรือแสดงออกถึงอัตลักษณ์ทางเพศของตัวเองโดยไม่โจมตีผู้อื่น
- ตัวอย่าง: "เพิ่งคุยกับพ่อแม่เรื่องที่เราเป็น bi ได้แล้ว โล่งใจมากเลย",
  "ไม่ได้รู้สึกตัวเองเป็นผู้หญิง 100% แต่ก็โอเคกับตัวเองนะ",
  "ใช้เวลานานมากกว่าจะยอมรับตัวเองได้ แต่ตอนนี้ดีขึ้นมากแล้ว"

ตัวอย่างเพิ่มเติมที่ต้องการความหลากหลายแบบนี้:
{{SEED_EXAMPLES}}

รูปแบบ JSON ที่ต้องการ:
{
  "items": [
    {
      "text": "ข้อความโซเชียลมีเดียภาษาไทย",
      "label": 2,
      "subtype": "counter_speech | gender_question | awareness | self_identification",
      "gender_ref": "เพศหรืออัตลักษณ์ที่กล่าวถึง (เช่น ผู้หญิง, ผู้ชาย, เกย์, LGBT+, ทั่วไป)"
    }
  ]
}

ส่งออกเฉพาะ JSON เท่านั้น ห้ามอธิบาย ห้ามใส่ markdown
จำนวน items ต้องเท่ากับ {{COUNT}} พอดี
"""

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = request.Request(url=url, data=data, method="POST", headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc}") from exc


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------


def call_openai(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = _post_json(url, headers, body)
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenAI response: {resp}") from exc


def call_gemini(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = _post_json(url, headers, body)
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {resp}") from exc


def call_openrouter(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call any model via OpenRouter (openrouter.ai) using the OpenAI-compatible API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Ask for JSON output; most OpenRouter models honour this
        "response_format": {"type": "json_object"},
    }
    resp = _post_json(url, headers, body, timeout=180)
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {resp}") from exc


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------


def _recover_items(raw: str) -> List[Dict[str, Any]]:
    """Best-effort recovery: parse individual {...} objects from a JSON array."""
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
                        items.append(json.loads(cleaned[obj_start : i + 1]))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
        elif ch == "]" and depth == 0:
            break
    return items


def parse_response(raw: str) -> List[Dict[str, Any]]:
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
# Generation loop
# ---------------------------------------------------------------------------

BATCH_SIZE = 25  # items per single API call (avoids token limits)


def generate_batch(
    *,
    mode: str,
    count: int,
    provider: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    if mode == "bias":
        system = BIAS_SYSTEM
        user_tmpl = BIAS_USER
    elif mode == "gender-discuss":
        system = GENDER_DISCUSS_SYSTEM
        user_tmpl = GENDER_DISCUSS_USER
    else:
        system = NEUTRAL_SYSTEM
        user_tmpl = NEUTRAL_USER

    user = user_tmpl.replace("{{COUNT}}", str(count))

    # Inject random seed examples for modes that use the {{SEED_EXAMPLES}} placeholder
    if "{{SEED_EXAMPLES}}" in user:
        if mode == "gender-discuss":
            pool = _GENDER_DISCUSS_SEED_POOL
        else:
            pool = _NEUTRAL_SEED_POOL
        sample_size = min(5, len(pool))
        seeds = random.sample(pool, sample_size)
        seed_str = "\n".join(f'- "{s}"' for s in seeds)
        user = user.replace("{{SEED_EXAMPLES}}", seed_str)

    if dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(system)
        print("\n=== USER PROMPT ===")
        print(user)
        return []

    if provider == "gemini":
        raw = call_gemini(
            api_key=api_key,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "openrouter":
        raw = call_openrouter(
            api_key=api_key,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raw = call_openai(
            api_key=api_key,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return parse_response(raw)


def _load_checkpoint(out_path: Path) -> List[Dict[str, Any]]:
    """Load already-collected items from an existing output or checkpoint file."""
    ckpt_path = out_path.with_suffix(".ckpt.json")
    for path in (ckpt_path, out_path):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    print(
                        f"  [resume] loaded {len(data)} items from {path.name}",
                        file=sys.stderr,
                    )
                    return data
            except (json.JSONDecodeError, OSError):
                pass
    return []


def _save_checkpoint(items: List[Dict[str, Any]], out_path: Path) -> None:
    """Atomically write progress to a checkpoint file after every batch."""
    ckpt_path = out_path.with_suffix(".ckpt.json")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(ckpt_path)


def _generate_batch_with_retry(
    *,
    batch_num: int,
    mode: str,
    count: int,
    provider: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    retry_delay: float,
    max_attempts: int = 5,
) -> List[Dict[str, Any]]:
    for attempt in range(max_attempts):
        try:
            return generate_batch(
                mode=mode,
                count=count,
                provider=provider,
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
                f"  [batch {batch_num}] error (attempt {attempt + 1}/{max_attempts}): {exc}",
                file=sys.stderr,
            )
            time.sleep(retry_delay * (attempt + 1))
    return []


def generate_all(
    *,
    mode: str,
    total: int,
    provider: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    out_path: Path,
    parallel_calls: int = 1,
    retry_delay: float = 5.0,
) -> List[Dict[str, Any]]:
    # Resume from checkpoint if available
    all_items: List[Dict[str, Any]] = _load_checkpoint(out_path) if not dry_run else []
    remaining = total - len(all_items)
    batch_num = 0

    if remaining <= 0:
        print(f"  [resume] already have {len(all_items)} items, nothing to do.", file=sys.stderr)
        return all_items[:total]

    if all_items:
        print(f"  [resume] continuing from {len(all_items)}/{total} …", file=sys.stderr)

    seen_texts = {item.get("text", "") for item in all_items}
    expected_label = 1 if mode == "bias" else (2 if mode == "gender-discuss" else 0)

    while remaining > 0:
        calls_this_wave = min(parallel_calls, (remaining + BATCH_SIZE - 1) // BATCH_SIZE)
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
                f"[batch {batch_num}] requesting {c} items (collected {len(all_items)}/{total})",
                file=sys.stderr,
            )

        wave_items: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=calls_this_wave) as executor:
            futures = [
                executor.submit(
                    _generate_batch_with_retry,
                    batch_num=bnum,
                    mode=mode,
                    count=bcount,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    dry_run=dry_run,
                    retry_delay=retry_delay,
                )
                for bnum, bcount in wave_batches
            ]
            for future in as_completed(futures):
                wave_items.extend(future.result())

        if dry_run:
            break

        # Deduplicate by text and validate label field.
        fresh = [i for i in wave_items if i.get("text", "") not in seen_texts]
        valid = [
            i for i in fresh if isinstance(i, dict) and i.get("text", "").strip() and i.get("label") == expected_label
        ]

        if not valid:
            print("  warning: no valid items recovered in this wave, retrying …", file=sys.stderr)
            time.sleep(retry_delay)
            continue

        all_items.extend(valid)
        seen_texts.update(i.get("text", "") for i in valid)
        remaining = total - len(all_items)

        # Save progress after every wave — never lose work again
        _save_checkpoint(all_items, out_path)
        print(f"  [checkpoint] {len(all_items)}/{total} saved", file=sys.stderr)

        if remaining > 0:
            time.sleep(1.0)  # polite pause between waves

    return all_items[:total]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Thai gender-bias (label=1) or neutral (label=0) social-media text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--bias",
        dest="mode",
        action="store_const",
        const="bias",
        help="Generate gender-biased text (label=1): sexism, gender_object, stereotypes",
    )
    mode_group.add_argument(
        "--neutral",
        dest="mode",
        action="store_const",
        const="neutral",
        help="Generate non-gender-biased text (label=0): may contain profanity unrelated to gender",
    )
    mode_group.add_argument(
        "--gender-discuss",
        dest="mode",
        action="store_const",
        const="gender-discuss",
        help="Generate gender-related but non-biased text (label=2): counter-speech, questions, awareness, self-identification",
    )

    parser.add_argument(
        "--provider",
        choices=["gemini", "gpt", "openrouter"],
        default="openrouter",
        help=(
            "API provider: 'gemini' (direct Google API), 'gpt' (direct OpenAI API), "
            "or 'openrouter' (route any model via openrouter.ai). Default: openrouter"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model ID override. "
            "Defaults: google/gemini-2.5-flash-001 (openrouter) | "
            "gemini-2.5-flash (gemini) | gpt-4o-mini (gpt)"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Total number of samples to generate. Default: 100",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=("Output JSON file path. Defaults: bias_samples.json or neutral_samples.json"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.95,
        help="Sampling temperature. Default: 0.95",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens per API call. Default: 8192",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=("API key override. Falls back to OPENROUTER_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY env vars."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts only, do not call the API.",
    )
    parser.add_argument(
        "--parallel-calls",
        type=int,
        default=1,
        help="Number of concurrent API batch calls. Default: 1",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    if args.parallel_calls < 1:
        print("Error: --parallel-calls must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Resolve defaults
    if args.model is None:
        if args.provider == "openrouter":
            args.model = "google/gemini-2.5-flash-001"
        elif args.provider == "gemini":
            args.model = "gemini-2.5-flash"
        else:
            args.model = "gpt-4o-mini"

    if args.output is None:
        if args.mode == "bias":
            args.output = "bias_samples.json"
        elif args.mode == "gender-discuss":
            args.output = "gender_discuss_samples.json"
        else:
            args.output = "neutral_samples.json"

    # Resolve API key
    api_key = args.api_key
    if not api_key:
        if args.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        elif args.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key and not args.dry_run:
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "gpt": "OPENAI_API_KEY",
        }
        env_var = env_map[args.provider]
        print(
            f"Error: no API key found. Set {env_var} in .env or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Mode: {args.mode} | Provider: {args.provider} | Model: {args.model} | "
        f"Count: {args.count} | Parallel: {args.parallel_calls} | Output: {args.output}",
        file=sys.stderr,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = generate_all(
        mode=args.mode,
        total=args.count,
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        out_path=out_path,
        parallel_calls=args.parallel_calls,
    )

    if args.dry_run:
        print("\n[dry-run] No output written.", file=sys.stderr)
        return

    out_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Remove checkpoint file now that final output is written
    ckpt_path = out_path.with_suffix(".ckpt.json")
    if ckpt_path.exists():
        ckpt_path.unlink()

    label_counts: Dict[Any, int] = {}
    subtype_counts: Dict[str, int] = {}
    for item in items:
        lbl = item.get("label")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        sub = item.get("subtype") or item.get("topic")
        if sub:
            subtype_counts[sub] = subtype_counts.get(sub, 0) + 1

    print(f"\nSaved {len(items)} items → {out_path}", file=sys.stderr)
    print(f"Label distribution: {label_counts}", file=sys.stderr)
    if subtype_counts:
        print(f"Subtype/topic distribution: {subtype_counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
