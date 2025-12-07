import html
import json
from pathlib import Path
import re

import pandas as pd
from pythainlp import word_tokenize
import pythainlp

from scraper.constants import OUTPUT_DIR

# THAI_TRAILING_VOWELS = {"า", "ิ", "ี", "ุ", "ู", "ั", "ื", "ึ", "็"}


def correct_floating_vowels(text):
    """
    Corrects floating following-vowel tokens in Thai text.
    
    :param text: The original Thai string (e.g., "อยาก เติม า ก").
    :return: The corrected string and the final tokens.
    """
    
    # 1. Define the set of following-vowels (must be accurate and complete)
    # Excludes leading vowels like 'เ', 'แ', 'โ', 'ใ', 'ไ'
    FOLLOWING_VOWELS = set('าิีึืุูะำ') 
    
    # Run initial tokenization
    initial_tokens = word_tokenize(text, engine="newmm")
    
    # Store replacements to be made in the original text
    replacements = []

    # 2. Iterative Correction
    for i in range(1, len(initial_tokens)):
        current_token = initial_tokens[i]
        prev_token = initial_tokens[i-1]
        
        # Check if the current token is a single following vowel
        if len(current_token) == 1 and current_token in FOLLOWING_VOWELS:
            
            # Get the last character of the preceding token
            consonant = prev_token[-1]
            
            # Construct the new merged token
            new_vowel_token = consonant + current_token # e.g., "ม" + "า" = "มา"
            
            # This is the tricky part: Finding the span in the original text
            # We look for the pattern: (prev_token)(any whitespace)(floating_vowel_token)
            # The replacement will only affect the floating_vowel_token part.
            
            # Use regex to find the *exact* floating vowel token after the preceding token.
            # We must escape the tokens for regex, and handle the intermediate spaces.
            # The pattern is designed to capture the exact floating token for replacement.
            
            # The original text search pattern: find the previous token, followed by 
            # any spaces, followed by the current floating vowel token.
            # We use a non-greedy capture group for the floating vowel token to ensure 
            # we only replace the specific instance we found in the token list.
            
            # Find the index of the start of the previous token.
            start_index_prev = text.find(prev_token) 
            if start_index_prev == -1: continue # Safety check
            
            # Start search *after* the previous token.
            search_from = start_index_prev + len(prev_token)
            
            # Regex to find the next occurrence of the floating vowel token
            # \s* ensures all spacing is accounted for
            pattern = re.escape(prev_token) + r'(\s*)' + re.escape(current_token)
            
            # Find the match in the text
            match = re.search(pattern, text)
            
            if match:
                # Group 1 is the whitespace(s) between the tokens
                whitespace = match.group(1) 
                
                # The replacement text for the floating vowel and its preceding space(s)
                # The goal is to replace (\s* + floating_vowel) with (\s* + new_vowel_token)
                
                # We save the needed replacement structure:
                # The string to find is: [whitespace] + [old floating token]
                find_str = whitespace + current_token
                # The string to replace with is: [whitespace] + [new merged token]
                replace_str = whitespace + new_vowel_token
                
                # We need to ensure we only replace the *specific* one found by the search.
                # Find the starting index of the 'find_str' *after* the previous token.
                # This ensures we don't accidentally replace a later floating vowel.
                # The full match span is from the start of prev_token to the end of the floating token
                start_match = match.start()
                end_match = match.end()
                
                # This replacement is hard: Let's use string splitting/joining for clarity
                # split the text at the PRECEDING token
                
                # Simplified approach: If there is *exactly one* occurrence of the pattern
                # in the rest of the string, it's safer.
                
                replacements.append({
                    'find': prev_token + whitespace + current_token,
                    'replace': prev_token + whitespace + new_vowel_token
                })

    # 3. Apply all text updates
    corrected_text = text
    for item in replacements:
        # NOTE: This uses re.sub but only for the first occurrence found
        # to ensure it's the specific instance we targeted. 
        # A more robust solution would track character offsets.
        
        # Example: Find "เติม า" and replace with "เติม มา"
        corrected_text = corrected_text.replace(item['find'], item['replace'], 1)
        
    
    return corrected_text

# ----------------------------------------
# Thai slang dictionary
# ----------------------------------------
THAI_SLANG_MAP = {
    "ผช": "ผู้ชาย",
    "ผช.": "ผู้ชาย",
    "ผญ": "ผู้หญิง",
    "ผญ.": "ผู้หญิง",
    "กระเทย": "กะเทย",

    "อส": "ไอสัส",
    "อห": "โอ้โห",

    # Education levels
    "ปวช": "ประกาศนียบัตรวิชาชีพ",
    "ปวส": "ประกาศนียบัตรวิชาชีพชั้นสูง",
    "มต้น": "มัธยมต้น",
    "มปลาย": "มัธยมปลาย",

    # Common Thai abbreviations
    "รร": "โรงเรียน",
    "นร": "นักเรียน",
    "นศ": "นักศึกษา",

    "เมนส์": "ประจำเดือน",
    "ผญค": "ผู้หญิงคน",
    "ผชค": "ผู้ชายคน",

    # Internet speech
    "คับ": "ครับ",
    "คั้บ": "ครับ",
    "ค้าบ": "ครับ",
    "คับผม": "ครับผม",
    "ครัช": "ครับ",
}

PUNCT_TAIL = ".,!?…~😂🤣555"


# ----------------------------------------
# Helper: Normalize Thai slang
# ----------------------------------------
def normalize_thai_slang(text: str) -> str:
    """Replace slang only when matching whole token-like patterns."""
    for slang, full in THAI_SLANG_MAP.items():
        text = re.sub(rf"(?<!\w){slang}(?!\w)", full, text)
    return text


# ----------------------------------------
# Main cleaning function
# ----------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # ---------- Basic cleaning ----------
    text = html.unescape(text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # URLs / mentions / hashtags
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\.[A-Za-z]+", " ", text)

    # ---------- Normalize BEFORE anything else ----------
    text = text.lower()
    text = pythainlp.util.normalize(text)

    # Collapse Thai repeated chars
    text = re.sub(r'([\u0E00-\u0E7F])\1{1,}', r'\1', text)

    # Collapse tone marks
    text = re.sub(r'([่้๊๋])\1+', r'\1', text)

    # Thai slang
    text = normalize_thai_slang(text)

    # Remove extra spaces (pass 1)
    text = re.sub(r"\s+", " ", text).strip()

    # ---------- Misc normalization ----------
    # ellipses
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"…+", "…", text)

    # quotes (')
    text = re.sub(r"[‘’‚‛´`]", "'", text)
    text = re.sub(r"'{2,}", "'", text)

    # quotes (")
    text = re.sub(r'[“”„‟]', '"', text)
    text = re.sub(r'"{2,}', '"', text)

    # ---------- Laugh normalization ----------
    text = re.sub(r"5{3,}\+*", "หัวเราะ", text)
    text = re.sub(r"(ฮ่า)+", "หัวเราะ", text)
    text = re.sub(r"(ฮา)+", "หัวเราะ", text)
    text = re.sub(r"(?i)ha(ha)+\+*", "หัวเราะ", text)
    text = re.sub(r"(หัวเราะ\s*){2,}", "หัวเราะ", text)
    text = re.sub(r"\d{5,}", "หัวเราะ", text)

    # ---------- Punctuation collapsing ----------
    text = re.sub(r"([!?])\1{1,}", r"\1", text)
    text = re.sub(r"([!?]){2,}", r"\1", text)

    # ---------- Allowed characters ----------
    text = re.sub(r"[^0-9A-Za-z\u0E00-\u0E7F\s\.\,\!\?\']", " ", text)

    # Final cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------------------
# Helper: Count Thai characters
# ----------------------------------------
def count_thai_characters(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"[\u0E00-\u0E7F]", text))


# ----------------------------------------
# Postprocess CSV
# ----------------------------------------
def postprocess_csv(
    input_path=Path(OUTPUT_DIR, "combined_output.csv"),
    output_path=Path(OUTPUT_DIR, "postprocessed_output.csv"),
    text_column="text",
):
    df = pd.read_csv(input_path)

    # Remove duplicates by id
    df = df.sort_values("id").drop_duplicates(subset=["id"], keep="first")

    # Remove duplicates by multiple columns
    compare_cols = [
        "platform", "platform_type", "url", "content_type",
        "timestamp", "scraper_module", "text", "source_file", "tokens"
    ]
    df = df.drop_duplicates(subset=compare_cols, keep="last")

    # Clean text
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    
    df[text_column] = df[text_column].apply(correct_floating_vowels)

    # Tokenization
    df["tokens"] = (
        df[text_column]
        .apply(lambda x: word_tokenize(str(x), engine="newmm"))
        .apply(lambda toks: [t for t in toks if t.strip() != ""])
    )

    # Thai char count
    df["thai_char_count"] = df[text_column].apply(count_thai_characters)

    # Filter
    df = df[df["thai_char_count"] >= 5]

    df = df.drop(columns=["thai_char_count"])
    df = df.drop(columns=["raw_text"], errors="ignore")

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned CSV → {output_path}")

    return df


if __name__ == "__main__":
    postprocess_csv()