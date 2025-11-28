from pathlib import Path
import ollama
import os
import time
import pandas as pd
import argparse

from prompteng.constants import PROMPT_DIR, OUTPUT_DIR, ASSETS_DIR

# from .constants import PROMPT_DIR, OUTPUT_DIR, ASSETS_DIR


# ----------------
# Helper
# ----------------
def read_file(path):
    if not os.path.exists(path):
        print(f"Error: Missing file {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ----------------
# Main
# ----------------
def main():
    parser = argparse.ArgumentParser(description="Run annotation with Ollama model.")
    parser.add_argument(
        "--model", type=str, default="qwen2.5:7b",
        help="Model name for Ollama (default: qwen2.5:7b)"
    )
    parser.add_argument(
        "--prompt", type=str, default="draft_1",
        help="Prompt folder name inside PROMPT_DIR"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Input CSV containing column 'tokens'"
    )
    parser.add_argument(
        "--output_csv", type=str, required=False,
        help="Output CSV to save results (default: OUTPUT_DIR/<input_name>_<model>.csv)"
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
        output_csv = OUTPUT_DIR / f"{input_name}_{safe_model}.csv"

    print(f"--- Starting Annotation ---")
    print(f"Model: {args.model}")
    print(f"Prompt folder: {prompt_folder}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {output_csv}")

    # Load prompts
    sys_prompt = read_file(system_file)
    user_template = read_file(user_template_file)
    if sys_prompt is None or user_template is None:
        raise FileNotFoundError("❌ System or user prompt missing.")

    # Load CSV dataset
    df = pd.read_csv(args.input_csv)
    if "tokens" not in df.columns:
        raise ValueError("❌ ERROR: CSV must contain column named 'tokens'")

    output_col = args.model
    df[output_col] = ""

    # iterate rows
    for i, row in df.iterrows():
        token_list = row["tokens"]

        # build final prompt
        prompt_filled = user_template.replace("<PASTE_TOKEN_LIST_HERE>", token_list)

        print(f"[{i+1}/{len(df)}] Processing...", end="\r")

        # LLM call
        try:
            response = ollama.chat(
                model=args.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_filled},
                ],
            )
            result = (
                response["message"]["content"]
                .replace("\n", " ")
                .replace("|", "")
            )
            df.at[i, output_col] = result

        except Exception as e:
            print(f"\nError on row {i}: {e}")
            df.at[i, output_col] = "ERROR"

        time.sleep(0.2)

    # save new CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n\n✅ Done! Saved to {output_csv}")


if __name__ == "__main__":
    main()