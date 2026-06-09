#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = ["id", "text", "predicted_label"]


def to_yaml_scalar(value: str, indent: int) -> str:
    if value is None:
        return "''"

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    if "\n" in text:
        pad = " " * indent
        lines = "\n".join(f"{pad}{line}" for line in text.split("\n"))
        return f"|-\n{lines}"

    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def convert_csv_to_yaml(input_csv: Path, output_yaml: Path, limit: int | None = None) -> int:
    rows_written = 0

    with input_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)

        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        with output_yaml.open("w", encoding="utf-8") as dst:
            dst.write("records:\n")
            for row in reader:
                if limit is not None and rows_written >= limit:
                    break

                row_id = row.get("id", "")
                row_text = row.get("text", "")
                row_label = row.get("predicted_label", "")

                dst.write(f"  - id: {to_yaml_scalar(row_id, indent=6)}\n")

                text_scalar = to_yaml_scalar(row_text, indent=6)
                if text_scalar.startswith("|-\n"):
                    dst.write("    text: |-\n")
                    dst.write(text_scalar[3:] + "\n")
                else:
                    dst.write(f"    text: {text_scalar}\n")

                dst.write(f"    predicted_label: {to_yaml_scalar(row_label, indent=6)}\n")
                rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert scraped_data.csv to YAML with only id, text, and predicted_label columns."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("services/auto_annalysis/assets/scraped_data.csv"),
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("services/auto_annalysis/assets/scraped_data.yaml"),
        help="Path to output YAML file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to export",
    )
    args = parser.parse_args()

    written = convert_csv_to_yaml(args.input, args.output, args.limit)
    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
