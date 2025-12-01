from pathlib import Path
import re
import pandas as pd
from pythainlp import word_tokenize

from scraper.constants import OUTPUT_DIR

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("\n", " ").replace("\r", " ")  # remove newlines
    s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
    return s.strip()

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