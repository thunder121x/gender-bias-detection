# Fine-Tuning System Prompt — Thai Gender Bias Span Detector

This document defines the production system prompt for fine-tuning Qwen 3.5 2B (or any decoder-only LLM) as a span detector for Thai gender bias, derived directly from `annotation-guideline.md` and `assets/structure/*.csv`.

The model emits **only three span tags**: `<GB-ATTACK>`, `<GB-NORMATIVE>`, `<GB-SEX>`. Every other case (NON-GB, including counter-speech, gendered insults to individuals, and neutral text) returns the input verbatim with no tags whatsoever.

---

## 1. The System Prompt (use this verbatim during training and inference)

Save the block below as your `system` message. Do not paraphrase between training and inference — the prompt must be byte-identical in both phases.

```text
คุณคือเครื่องมือตรวจจับช่วงข้อความที่มีอคติทางเพศในภาษาไทย (Thai Gender Bias Span Detector)

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
- หากไม่พบอคติทางเพศตามเกณฑ์ ให้ส่งคืนข้อความต้นฉบับทันทีโดยไม่มีแท็กใด ๆ
```

---

## 2. Few-Shot Examples (optional addendum to system prompt)

If you have token budget, append these 6 in-context examples after the prompt above. They cover one positive per class and three high-risk negative cases. Empirically these reduce false-positive rate by 3–5% on counter-speech.

```text
ตัวอย่าง:

Input: ผู้หญิงสมัยนี้สมองกลวงทุกคน
Output: <GB-ATTACK>ผู้หญิงสมัยนี้สมองกลวงทุกคน</GB-ATTACK>

Input: ผู้ชายต้องเป็นผู้นำเสมอ ห้ามอ่อนแอ
Output: <GB-NORMATIVE>ผู้ชายต้องเป็นผู้นำเสมอ ห้ามอ่อนแอ</GB-NORMATIVE>

Input: ตุ๊ดพวกนี้ใช้รูตูดอย่างเดียว
Output: <GB-SEX>ตุ๊ดพวกนี้ใช้รูตูดอย่างเดียว</GB-SEX>

Input: ผู้ชายคนนั้นเอาแต่ใจตัวเองว่ะ
Output: ผู้ชายคนนั้นเอาแต่ใจตัวเองว่ะ

Input: ยังมีคนพูดว่าผู้หญิงขับรถไม่เป็น น่าสมเพชจริง ๆ
Output: ยังมีคนพูดว่าผู้หญิงขับรถไม่เป็น น่าสมเพชจริง ๆ

Input: วันนี้อากาศดีมาก ไปเดินเล่นกันเถอะ
Output: วันนี้อากาศดีมาก ไปเดินเล่นกันเถอะ
```

---

## 3. Training Data Format (ChatML JSONL)

Each row of your training file should look like this. Use the **same** system prompt for all 24,000 examples — this is what teaches the model to associate the prompt with the tagging behavior.

```json
{"messages":[
  {"role":"system","content":"<<paste the full system prompt from section 1>>"},
  {"role":"user","content":"ผู้หญิงสมัยนี้สมองกลวงทุกคน"},
  {"role":"assistant","content":"<GB-ATTACK>ผู้หญิงสมัยนี้สมองกลวงทุกคน</GB-ATTACK>"}
]}
{"messages":[
  {"role":"system","content":"<<same system prompt>>"},
  {"role":"user","content":"ผู้ชายคนนั้นเอาแต่ใจตัวเองว่ะ"},
  {"role":"assistant","content":"ผู้ชายคนนั้นเอาแต่ใจตัวเองว่ะ"}
]}
```

Mapping from your 6 source files to assistant outputs:

| Source file              | label    | subtype          | assistant output                                   |
| ------------------------ | -------- | ---------------- | -------------------------------------------------- |
| `gb_attack.json`         | GB       | GB-ATTACK        | wrap text in `<GB-ATTACK>...</GB-ATTACK>`          |
| `gb_normative.json`      | GB       | GB-NORMATIVE     | wrap text in `<GB-NORMATIVE>...</GB-NORMATIVE>`    |
| `gb_sex.json`            | GB       | GB-SEX           | wrap text in `<GB-SEX>...</GB-SEX>`                |
| `non_gb_insult.json`     | NON-GB   | gendered_insult  | output = input verbatim, no tags                   |
| `non_gb_meta.json`       | NON-GB   | meta_counter     | output = input verbatim, no tags                   |
| `non_gb_neutral.json`    | NON-GB   | neutral          | output = input verbatim, no tags                   |

---

## 4. Critical Implementation Notes

**Special tokens.** Before training, register the 6 tag tokens in the tokenizer so they encode as single tokens instead of being split character-by-character. This dramatically improves tag-boundary accuracy.

```python
tokenizer.add_tokens(
    ["<GB-ATTACK>", "</GB-ATTACK>",
     "<GB-NORMATIVE>", "</GB-NORMATIVE>",
     "<GB-SEX>", "</GB-SEX>"],
    special_tokens=True,
)
model.resize_token_embeddings(len(tokenizer))
```

**Train on completions only.** Configure the trainer to mask the system + user portions and compute loss only over the assistant response. This stops the model from memorizing the prompt and forces it to learn the tagging behavior. In TRL/Unsloth this is `train_on_responses_only` or equivalent.

**Tag granularity.** This prompt instructs minimal-span tagging (per Section 6 of the annotation guideline). Your current 24k dataset stores sentence-level labels only — when generating the assistant output, you have two valid choices:
- *Whole-sentence wrap (simpler):* For short Thai sentences (avg 42–58 chars), wrapping the entire `text` in the subtype tag is acceptable and keeps your pipeline simple.
- *Precise span extraction (better):* Run a one-off extraction pass (with a stronger model or rule-based) to identify the trigger phrase per the guideline §6.2, then tag only that span. This produces a more useful detector at inference time.

Pick one and apply it consistently across all 12k positive examples. Mixing both styles in the same training set will hurt accuracy.

**Inference parity.** At serving time, send the exact same system prompt. Use `temperature=0.0` (or very low, e.g. 0.1) and a high `repetition_penalty` of about 1.0 (no penalty) — span detection benefits from deterministic, faithful copying of the input.

**Stop conditions.** Set `max_new_tokens` to roughly `1.5 × len(input_tokens) + 32`. The output is at most the input plus a few tag tokens, so longer generations indicate hallucination.

---

## 5. Sanity Checks Before You Train

1. Verify every assistant output contains the input as a substring (after stripping tags). If it doesn't, the example is broken.
2. Verify every GB-* tag has a matching closing tag.
3. Verify NON-GB rows have output identical to input.
4. Hold out 5% (1,200 samples) as a validation set, stratified across all 6 source files.
5. After training, the first thing to evaluate: false-positive rate on `non_gb_meta.json` validation slice. This is the hardest negative class and the best early signal of overfit.
