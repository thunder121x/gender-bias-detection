#!/usr/bin/env python3
"""
System Prompt Validation Tool
Ensures byte-identical system prompt across all files.
"""

import hashlib
from pathlib import Path

# PRODUCTION SYSTEM PROMPT (MUST BE IDENTICAL)
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


def validate_system_prompt():
    """Validate system prompt across all files"""
    print("=" * 80)
    print("SYSTEM PROMPT VALIDATION")
    print("=" * 80)
    
    # Calculate hash
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    prompt_lines = len(SYSTEM_PROMPT.split('\n'))
    prompt_chars = len(SYSTEM_PROMPT)
    
    print(f"\nReference System Prompt:")
    print(f"  SHA256: {prompt_hash}")
    print(f"  Lines: {prompt_lines}")
    print(f"  Characters: {prompt_chars}")
    
    # Files to check (only files that use SYSTEM_PROMPT variable)
    # Note: finetune_qwen_span_detector.py and finetune_qwen_lora.py embed the prompt in JSONL data
    files_to_check = [
        ("finetune_qwen_span_detector.py", "SYSTEM_PROMPT = "),  # Used to generate training data
        ("inference_qwen_span.py", "SYSTEM_PROMPT = "),           # Used for inference
    ]
    
    print(f"\nValidating {len(files_to_check)} files...\n")
    
    all_valid = True
    
    for file_path, search_str in files_to_check:
        full_path = Path(__file__).resolve().parent / file_path
        
        if not full_path.exists():
            print(f"❌ {file_path}: FILE NOT FOUND")
            all_valid = False
            continue
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract system prompt from file
        try:
            start_idx = content.find(search_str) + len(search_str)
            
            # Find the end (either another variable or end of triple quote)
            end_idx = content.find('"""', start_idx + 100)
            if end_idx == -1:
                end_idx = content.find("'''", start_idx + 100)
            
            file_prompt = content[start_idx:end_idx].strip()
            
            # Remove quote characters if present
            if file_prompt.startswith('"""') or file_prompt.startswith("'''"):
                file_prompt = file_prompt[3:]
            
            file_hash = hashlib.sha256(file_prompt.encode()).hexdigest()
            
            if file_hash == prompt_hash:
                print(f"✅ {file_path}: VALID (SHA256 matches)")
            else:
                print(f"❌ {file_path}: MISMATCH")
                print(f"   Expected: {prompt_hash}")
                print(f"   Got:      {file_hash}")
                all_valid = False
        
        except Exception as e:
            print(f"⚠️  {file_path}: Could not extract prompt - {e}")
    
    print("\n" + "=" * 80)
    
    if all_valid:
        print("✅ ALL SYSTEM PROMPTS ARE BYTE-IDENTICAL")
        print("\nTraining and inference will use the same prompt.")
    else:
        print("❌ SYSTEM PROMPT MISMATCH DETECTED")
        print("\nThis will cause training/inference discrepancy!")
        print("Please fix the prompts before training.")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    valid = validate_system_prompt()
    sys.exit(0 if valid else 1)
