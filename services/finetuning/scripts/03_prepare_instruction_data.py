#!/usr/bin/env python3
"""
Script to prepare instruction-input-output format training data for LoRA fine-tuning.
Converts instruction data to ChatML format for training.

Input format:
{
  "instruction": "Identify and tag social bias in the following Thai text...",
  "input": "ผู้หญิงทุกคนก็โง่ สวัสดี",
  "output": "<GB-NORM>ผู้หญิงทุกคนก็โง่</GB-NORM>สวัสดี"
}

Output format (ChatML):
{
  "text": "<s>[INST] <<SYS>>\nYou are a helpful assistant...\n<</SYS>>\n\nInstruction: Identify and tag...\nInput: ผู้หญิงทุกคนก็โง่ สวัสดี\n[/INST] <GB-NORM>ผู้หญิงทุกคนก็โง่</GB-NORM>สวัสดี</s>"
}

Usage:
    python scripts/prepare_instruction_data.py --input instruction_train.jsonl --output train_chatml.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


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


class InstructionToChatML:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def convert_sample(self, sample: Dict) -> Dict:
        """Convert instruction-input-output format to ChatML format."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        
        # Combine instruction and input
        prompt = f"{instruction}\n\nInput: {input_text}"
        
        # Create ChatML format
        chatml_text = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{prompt}\n[/INST] {output_text}</s>"
        
        return {
            "text": chatml_text,
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }

    def process_file(self, input_file: str, output_file: str):
        """Convert instruction data file to ChatML format."""
        print(f"Reading from: {input_file}")
        print(f"Writing to: {output_file}")
        
        count = 0
        skipped = 0
        
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:
            
            for line in infile:
                try:
                    sample = json.loads(line)
                    converted = self.convert_sample(sample)
                    outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
                    
                    if (count + skipped) % 1000 == 0:
                        print(f"  Processed {count + skipped} samples ({count} valid, {skipped} skipped)")
                
                except json.JSONDecodeError as e:
                    skipped += 1
                    print(f"Warning: Skipped invalid JSON: {e}")
                except Exception as e:
                    skipped += 1
                    print(f"Warning: Skipped sample: {e}")
        
        print(f"\nConversion complete!")
        print(f"  Total processed: {count + skipped}")
        print(f"  Valid samples: {count}")
        print(f"  Skipped: {skipped}")
        
        return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert instruction-input-output format to ChatML for training"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file (instruction format)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file (ChatML format)",
    )
    parser.add_argument(
        "--system-prompt",
        help="Custom system prompt (optional)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    converter = InstructionToChatML(
        system_prompt=args.system_prompt or SYSTEM_PROMPT
    )
    
    converter.process_file(args.input, args.output)


if __name__ == "__main__":
    main()
