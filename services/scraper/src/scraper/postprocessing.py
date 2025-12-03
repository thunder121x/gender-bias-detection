import html
import json
from pathlib import Path
import re
import pandas as pd
from pythainlp import word_tokenize

from scraper.constants import OUTPUT_DIR

def clean_text(text: str) -> str:
    """Full cleaning + preprocessing:
    - handle None / non-string
    - unescape HTML
    - remove URLs / mentions / hashtags
    - normalize whitespace + newlines
    - lowercase English
    - collapse ellipses (...)
    - normalize laughing patterns (555, ฮ่าๆๆ, hahaha)
    - convert large numbers → 'หัวเราะ'
    - keep emojis (ไม่ลบ)
    - keep only Thai/Eng digits symbols
    """
    if not isinstance(text, str):
        return ""

    # ---------- Base cleaning ----------
    text = html.unescape(text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # URLs / mentions / hashtags
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)

    # collapse multi spaces (first pass)
    text = re.sub(r"\s+", " ", text).strip()

    # ---------- Normalization ----------
    text = text.lower()   # English only, Thai unaffected

    # ellipsis collapse: ... → ... | … → …
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"…+", "…", text)

    # ---------- Laugh detection ----------
    # 55555++++
    text = re.sub(r"5{3,}\+*", "หัวเราะ", text)

    # ฮ่าๆๆ / ฮาๆๆ
    text = re.sub(r"(ฮ่า)+", "หัวเราะ", text)
    text = re.sub(r"(ฮา)+", "หัวเราะ", text)

    # hahaha / hahahaha+++ (case-insensitive)
    text = re.sub(r"(?i)ha(ha)+\+*", "หัวเราะ", text)

    # collapse multiple "หัวเราะหัวเราะหัวเราะ"
    text = re.sub(r"(หัวเราะ)+", "หัวเราะ", text)

    # large numbers → หัวเราะ (behaviour จาก preprocess เดิม)
    text = re.sub(r"\d{5,}", "หัวเราะ", text)

    # ---------- Punctuation collapsing ----------
    # !!! → !, ??? → ?, !!?? → ?
    text = re.sub(r"([!?])\1{1,}", r"\1", text)
    text = re.sub(r"([!?]){2,}", r"\1", text)

    # ---------- Allowed characters ----------
    # keep Thai (U+0E00–0E7F), English, numbers, punctuation, emoji allowed
    text = re.sub(r"[^0-9A-Za-z\u0E00-\u0E7F\s\.\,\!\?\']", " ", text)

    # ---------- Final cleanup ----------
    text = re.sub(r"\s+", " ", text).strip()
    return text

def count_thai_characters(text):
    """Count Thai characters in string."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"[\u0E00-\u0E7F]", text))

def postprocess_csv(input_path=Path(OUTPUT_DIR,"combined_output.csv"), output_path=Path(OUTPUT_DIR,"postprocessed_output.csv"), text_column="text"):
    """
    - Strip whitespace
    - Count Thai characters
    - Filter out rows with < 5 Thai characters
    """
    df = pd.read_csv(input_path)

    # Clean text: remove newlines + extra spaces
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    df["tokens"] = (
        df[text_column]
            .apply(lambda x: word_tokenize(str(x), engine="newmm"))
            .apply(lambda toks: [t for t in toks if t.strip() != ""])
            # .apply(lambda t: json.dumps(t, ensure_ascii=False))
    )
    # Count Thai characters
    df["thai_char_count"] = df[text_column].apply(count_thai_characters)

    # Filter Thai chars < 5
    df = df[df["thai_char_count"] >= 5]

    df = df.drop(columns=["thai_char_count"])
    df = df.drop(columns=["raw_text"])

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned CSV → {output_path}")

    return df


if __name__ == "__main__":
    postprocess_csv()