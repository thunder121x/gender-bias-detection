import argparse
import io
import json
import os
import time
from pathlib import Path

import ollama
import pandas as pd

from prompteng.constants import PROMPT_DIR, OUTPUT_DIR
# NOTE: Replace postprocess_json_output with the CSV version below
# from prompteng.postprocessing import postprocess_json_output

BATCH_SAVE = 1


# ----------------
# Helper
# ----------------
def read_file(path: Path) -> str | None:
    if not path.exists():
        print(f"Error: Missing file {path}")
        return None
    return path.read_text(encoding="utf-8").strip()


def normalize_tokens_cell(token_cell) -> str:
    """
    Ensure tokens is a valid JSON array string.
    Accepts:
      - list -> dumps to JSON
      - str JSON -> loads then dumps to normalized JSON
      - other -> str()
    """
    if isinstance(token_cell, list):
        return json.dumps(token_cell, ensure_ascii=False)

    s = str(token_cell)
    # Try to normalize if it's JSON-like
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return s


def postprocess_csv_output(raw: str) -> str:
    """
    Keep only CSV text (prefer header+one-row).
    - strips markdown fences
    - removes extra commentary lines
    """
    s = (raw or "").strip()

    # Remove markdown fences if present
    if s.startswith("```"):
        # drop first fence line
        s = s.split("\n", 1)[1] if "\n" in s else ""
        # drop trailing fence
        if "```" in s:
            s = s.rsplit("```", 1)[0].strip()

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]

    # Keep lines that look like CSV (contain comma)
    csv_lines = [ln for ln in lines if "," in ln]

    # Expect at least header + one data row
    if len(csv_lines) >= 2:
        return "\n".join(csv_lines[:2]).strip()

    # Fallback: return original stripped content
    return s


def parse_one_row_reduced_csv(csv_text: str) -> dict:
    """
    Parse a 2-line reduced CSV (header + one row) into a dict.
    """
    try:
        out_df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise ValueError(f"Cannot parse model CSV output: {e}\n---RAW---\n{csv_text}")

    if len(out_df) != 1:
        raise ValueError(f"Expected exactly 1 output row, got {len(out_df)}\n---RAW---\n{csv_text}")

    row = out_df.iloc[0].to_dict()
    return row


def build_one_row_input_csv(row: pd.Series, columns: list[str]) -> str:
    """
    Build a one-row CSV string (header + row) to feed into the model.
    Ensures tokens is normalized JSON array string.
    """
    data = {}
    for col in columns:
        if col == "tokens":
            data[col] = normalize_tokens_cell(row[col])
        else:
            data[col] = "" if pd.isna(row.get(col, "")) else row.get(col, "")

    one_df = pd.DataFrame([data])
    return one_df.to_csv(index=False)


# ----------------
# Main
# ----------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gender Bias/SOGI annotation with Ollama model.")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Model name for Ollama (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="draft_1",
        help="Prompt folder name inside PROMPT_DIR",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV containing at least columns 'id' and 'tokens'",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=False,
        help="Output CSV to save results (default: OUTPUT_DIR/<input_name>_<model>_wa.csv)",
    )
    parser.add_argument(
        "--autosave",
        action="store_true",
        help=f"Enable autosave checkpoint every {BATCH_SAVE} rows (default: OFF)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between requests (default: 0.2)",
    )
    args = parser.parse_args()

    # Resolve prompt folder
    prompt_folder = PROMPT_DIR / args.prompt
    system_file = prompt_folder / "system_prompt.txt"
    user_template_file = prompt_folder / "user_prompt.txt"

    # Default output CSV if not provided
    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        input_name = Path(args.input_csv).stem
        safe_model = args.model.replace(":", "_")
        output_csv = OUTPUT_DIR / f"{input_name}_{safe_model}_wa.csv"

    autosave_path = output_csv.with_name(output_csv.stem + "_autosave.csv")

    print("--- Starting Annotation ---")
    print(f"Model: {args.model}")
    print(f"Prompt folder: {prompt_folder}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {output_csv}")
    print("Autosave:", "ON" if args.autosave else "OFF")

    # Load prompts
    sys_prompt = read_file(system_file)
    user_template = read_file(user_template_file)
    if sys_prompt is None or user_template is None:
        raise FileNotFoundError("❌ System or user prompt missing.")

    # Load CSV dataset
    df = pd.read_csv(args.input_csv)

    # Ensure required columns exist
    if "id" not in df.columns:
        raise ValueError("❌ ERROR: CSV must contain column named 'id'")
    if "tokens" not in df.columns:
        raise ValueError("❌ ERROR: CSV must contain column named 'tokens'")

    # Columns to pass into model (include text if present)
    input_cols = ["id", "tokens"]
    if "text" in df.columns:
        input_cols.append("text")

    # Prepare output columns
    wa_cols = ["wa_rationales", "wa_triggers", "wa_label_type", "wa_decision_rule", "wa_binary"]
    for c in wa_cols:
        if c not in df.columns:
            df[c] = ""

    # iterate rows
    for i, row in df.iterrows():
        one_row_csv = build_one_row_input_csv(row, input_cols)

        # build final prompt
        prompt_filled = user_template.replace("<PASTE_INPUT_CSV_HERE>", one_row_csv)

        print(f"[{i + 1}/{len(df)}] Processing...", end="\r")

        # LLM call
        try:
            response = ollama.chat(
                model=args.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_filled},
                ],
            )
            raw = response["message"]["content"]

            cleaned_csv = postprocess_csv_output(raw)
            out_row = parse_one_row_reduced_csv(cleaned_csv)

            # Validate id consistency (best-effort)
            if str(out_row.get("id", "")).strip() != str(row["id"]).strip():
                # Still write but warn
                print(f"\n⚠ Warning: Output id mismatch at row {i}. in={row['id']} out={out_row.get('id')}")

            # Write outputs
            for c in wa_cols:
                df.at[i, c] = out_row.get(c, "")

        except Exception as e:
            print(f"\n❌ Error on row {i} (id={row.get('id')}): {e}")
            for c in wa_cols:
                df.at[i, c] = "ERROR" if c == "wa_binary" else ""

        time.sleep(args.sleep)

        if args.autosave and (i + 1) % BATCH_SAVE == 0:
            df[["id"] + wa_cols].to_csv(autosave_path, index=False, encoding="utf-8-sig")
            print(f"\n💾 Autosaved checkpoint at row {i + 1}: {autosave_path}")

    # save reduced CSV
    reduced_df = df[["id"] + wa_cols]
    reduced_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n\n✅ Done! Saved to {output_csv}")

    # Remove autosave file if exists
    if args.autosave and autosave_path.exists():
        try:
            os.remove(autosave_path)
            print(f"🧹 Removed autosave file: {autosave_path}")
        except Exception:
            print(f"⚠ Could not delete autosave file: {autosave_path}")


if __name__ == "__main__":
    main()
