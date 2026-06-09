"""Utility functions for auto-analysis service."""

import yaml
from pathlib import Path
from typing import Any, Dict, List


def load_yaml(file_path: Path) -> Any:
    """Load YAML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, file_path: Path) -> None:
    """Save data to YAML file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def load_guideline(guideline_path: Path) -> str:
    """Load annotation guideline."""
    with open(guideline_path, "r", encoding="utf-8") as f:
        return f.read()


def load_scraped_data(data_path: Path) -> List[Dict]:
    """Load scraped data and return records."""
    data = load_yaml(data_path)
    return data.get("records", [])


def chunk_records(records: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Split records into batches."""
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def format_batch_for_api(records: List[Dict]) -> str:
    """Format batch of records for Gemini API request."""
    formatted = "Please validate the following annotated records against the Gender Bias annotation guidelines.\n\n"
    formatted += "RECORDS TO VALIDATE:\n"
    formatted += "=" * 80 + "\n\n"

    for i, record in enumerate(records, 1):
        formatted += f"{i}. ID: {record['id']}\n"
        formatted += f"   Text: {record['text']}\n"
        formatted += f"   Predicted Label: {record['predicted_label']}\n\n"

    return formatted
