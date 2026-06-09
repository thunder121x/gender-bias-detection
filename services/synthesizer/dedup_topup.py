#!/usr/bin/env python3
"""
dedup_topup.py — Remove intra-file duplicate texts and top up to TARGET_COUNT.

Usage:
    python services/synthesizer/dedup_topup.py

Reads GEMINI_API_KEY from .env (cwd) or services/synthesizer/.env.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: load .env and make the package importable
# ---------------------------------------------------------------------------

_CWD_ENV = Path.cwd() / ".env"
_PKG_ENV = Path(__file__).parent / ".env"

def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)

_load_env(_CWD_ENV)
_load_env(_PKG_ENV)

# Make package importable without `pip install -e`
_src = Path(__file__).parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from synthesizer_v2.constants import DEFAULT_MODEL, OUTPUT_DIR
from synthesizer_v2.generate import generate_all

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_COUNT = 4000

MODE_TO_FILE: dict[str, str] = {
    "gb-attack":      "gb_attack.json",
    "gb-normative":   "gb_normative.json",
    "gb-sex":         "gb_sex.json",
    "non-gb-neutral": "non_gb_neutral.json",
    "non-gb-meta":    "non_gb_meta.json",
    "non-gb-insult":  "non_gb_insult.json",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def dedup_file(path: Path) -> tuple[list[dict], int]:
    """Load JSON, deduplicate by text, return (deduped_list, n_removed)."""
    data: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in data:
        t = item.get("text", "")
        if t not in seen:
            seen.add(t)
            deduped.append(item)
    return deduped, len(data) - len(deduped)


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    for mode, fname in MODE_TO_FILE.items():
        out_path = OUTPUT_DIR / fname
        if not out_path.exists():
            print(f"[{mode}] file not found: {out_path}, skipping", file=sys.stderr)
            continue

        deduped, n_removed = dedup_file(out_path)
        print(f"[{mode}] {fname}: {len(deduped) + n_removed} → {len(deduped)} after dedup ({n_removed} removed)")

        if n_removed == 0 and len(deduped) >= TARGET_COUNT:
            print(f"[{mode}] already clean at {len(deduped)}, nothing to do.")
            continue

        # Save deduped version so generate_all resumes from here (it uses seen_texts)
        out_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")

        topup_needed = TARGET_COUNT - len(deduped)
        if topup_needed <= 0:
            print(f"[{mode}] {len(deduped)} >= {TARGET_COUNT}, trimming to target.")
            final = deduped[:TARGET_COUNT]
            out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        print(f"[{mode}] need {topup_needed} more samples, generating …")
        items = generate_all(
            mode=mode,
            total=TARGET_COUNT,
            api_key=api_key,
            model=DEFAULT_MODEL,
            temperature=0.95,
            max_tokens=8192,
            dry_run=False,
            out_path=out_path,
            parallel=3,
        )

        final = items[:TARGET_COUNT]
        out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

        # Verify
        texts = [i.get("text", "") for i in final]
        remaining_dupes = len(texts) - len(set(texts))
        print(f"[{mode}] DONE: {len(final)} items, {remaining_dupes} dupes remaining")

    print("\nAll modes processed.")


if __name__ == "__main__":
    main()
