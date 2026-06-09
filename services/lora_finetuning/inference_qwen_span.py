#!/usr/bin/env python3
"""
Thai Gender Bias Span Detector - Inference Service
Production-grade inference with exact system prompt parity.

Key requirements:
- temperature=0.0 (deterministic)
- max_new_tokens = 1.5 * len(input_tokens) + 32
- Stop at 3 closing tags
- No hallucination
"""

import os
import sys
import json
import argparse
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# ============================================================================
# PRODUCTION SYSTEM PROMPT (MUST BE BYTE-IDENTICAL TO TRAINING)
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
# MODEL LOADER & INFERENCE ENGINE
# ============================================================================

class GenderBiasDetector:
    """Inference engine for Thai Gender Bias Span Detection"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading model from: {model_path}")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model in inference mode
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    def detect(self, text: str, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Detect gender bias spans in Thai text.
        
        Args:
            text: Thai text input
            temperature: 0.0 for deterministic output
            
        Returns:
            Dict with:
            - output: Tagged text
            - has_bias: bool
            - tags_found: List of tag types
        """
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        
        # Tokenize
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Calculate max tokens
        input_len = input_ids.shape[1]
        max_new_tokens = min(int(input_len * 1.5) + 32, 2048)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,  # Deterministic
                top_p=1.0,
                do_sample=temperature > 0,
                repetition_penalty=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=False
        )
        
        # Extract output (remove assistant label if present)
        output = response.strip()
        if output.startswith("Assistant:"):
            output = output[len("Assistant:"):].strip()
        
        # Parse tags
        tags_found = self._extract_tags(output)
        has_bias = len(tags_found) > 0
        
        return {
            "output": output,
            "has_bias": has_bias,
            "tags_found": tags_found,
            "input": text,
        }
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tag types from output"""
        tags = []
        
        if "<GB-ATTACK>" in text:
            tags.append("GB-ATTACK")
        if "<GB-NORMATIVE>" in text:
            tags.append("GB-NORMATIVE")
        if "<GB-SEX>" in text:
            tags.append("GB-SEX")
        
        return tags


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_mode(detector: GenderBiasDetector):
    """Run interactive detection loop"""
    print("\n" + "=" * 80)
    print("THAI GENDER BIAS SPAN DETECTOR - INTERACTIVE MODE")
    print("=" * 80)
    print("\nEnter Thai text to analyze (or 'quit' to exit):\n")
    
    while True:
        try:
            text = input(">>> ").strip()
            
            if text.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            result = detector.detect(text)
            
            print(f"\n📝 Input: {result['input']}")
            print(f"📌 Output: {result['output']}")
            print(f"⚠️  Has Bias: {result['has_bias']}")
            
            if result['tags_found']:
                print(f"🏷️  Tags: {', '.join(result['tags_found'])}")
            
            print()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(detector: GenderBiasDetector, input_file: str, output_file: str):
    """Process batch of texts from JSONL file"""
    print(f"\nProcessing batch from: {input_file}")
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
                text = item.get("text", "")
                
                if not text:
                    continue
                
                result = detector.detect(text)
                result["original_item"] = item
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1} items...")
            
            except Exception as e:
                print(f"Error on line {i}: {e}")
                continue
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Results saved to: {output_file}")
    
    # Statistics
    bias_count = sum(1 for r in results if r['has_bias'])
    print(f"\nStatistics:")
    print(f"  Total: {len(results)}")
    print(f"  With Bias: {bias_count} ({100 * bias_count / len(results):.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Thai Gender Bias Span Detector - Inference"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "qwen_gb_detector_lora"),
        help="Path to fine-tuned model"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Run mode"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file for batch mode (JSONL)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for batch mode"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load model
    detector = GenderBiasDetector(args.model_path, device=args.device)
    
    # Run mode
    if args.mode == "interactive":
        interactive_mode(detector)
    else:
        if not args.input or not args.output:
            print("Error: --input and --output required for batch mode")
            sys.exit(1)
        
        batch_mode(detector, args.input, args.output)


if __name__ == "__main__":
    main()
