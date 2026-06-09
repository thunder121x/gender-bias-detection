"""
Build labeled.yaml and compute confusion matrix.

Steps:
1. Load incorrect_items.yaml -> build {id: correct_label} mapping
2. Load scraped_data.yaml -> for each record, replace predicted_label with
   correct_label if available, else keep predicted_label (i.e. it was correct)
3. Save labeled.yaml (same structure as scraped_data.yaml but with true labels)
4. Compute and print confusion matrix (true=labeled, pred=scraped_data)
"""

import yaml
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent

INCORRECT_PATH  = BASE / "output" / "incorrect_items.yaml"
SCRAPED_PATH    = BASE / "assets"  / "scraped_data.yaml"
LABELED_PATH    = BASE / "output"  / "labeled.yaml"


# ── 1. Load corrections ────────────────────────────────────────────────────────
print("Loading incorrect_items.yaml …", flush=True)
with INCORRECT_PATH.open(encoding="utf-8") as f:
    incorrect_data = yaml.safe_load(f)

corrections: dict[str, str] = {}
skipped = 0
for item in incorrect_data["records"]:
    if "correct_label" in item:
        corrections[item["id"]] = item["correct_label"]
    else:
        skipped += 1
print(f"  {len(corrections):,} corrected items loaded ({skipped} skipped — missing correct_label).")


# ── 2. Load scraped data ───────────────────────────────────────────────────────
print("Loading scraped_data.yaml …", flush=True)
with SCRAPED_PATH.open(encoding="utf-8") as f:
    scraped_data = yaml.safe_load(f)

records = scraped_data["records"]
print(f"  {len(records):,} records loaded.")


# ── 3. Build labeled records & collect labels for confusion matrix ─────────────
print("Building labeled.yaml …", flush=True)
labeled_records = []
y_true: list[str] = []
y_pred: list[str] = []

for rec in records:
    rid   = rec["id"]
    pred  = rec["predicted_label"]
    true  = corrections.get(rid, pred)   # correct_label if wrong, else same as pred

    labeled_records.append({"id": rid, "text": rec["text"], "label": true})
    y_true.append(true)
    y_pred.append(pred)


# ── 4. Save labeled.yaml ───────────────────────────────────────────────────────
print(f"Saving {LABELED_PATH} …", flush=True)
with LABELED_PATH.open("w", encoding="utf-8") as f:
    yaml.dump({"records": labeled_records}, f, allow_unicode=True, sort_keys=False)
print(f"  Saved {len(labeled_records):,} records.")


# ── 5. Confusion matrix ────────────────────────────────────────────────────────
print("\nComputing confusion matrix …", flush=True)

all_classes = sorted(set(y_true) | set(y_pred))
n = len(all_classes)
cls_idx = {c: i for i, c in enumerate(all_classes)}

# matrix[true][pred]
matrix = [[0] * n for _ in range(n)]
for t, p in zip(y_true, y_pred):
    matrix[cls_idx[t]][cls_idx[p]] += 1


# ── 6. Pretty-print ────────────────────────────────────────────────────────────
COL_W = max(len(c) for c in all_classes) + 2
ROW_W = COL_W

header = " " * ROW_W + "".join(f"{c:>{COL_W}}" for c in all_classes) + "  (predicted)"
sep    = "-" * len(header)

print()
print("CONFUSION MATRIX  (rows = true label, cols = predicted label)")
print(sep)
print(header)
print(sep)
for i, cls in enumerate(all_classes):
    row_str = f"{cls:<{ROW_W}}" + "".join(f"{matrix[i][j]:>{COL_W}}" for j in range(n))
    print(row_str)
print(sep)

# ── 7. Per-class metrics ───────────────────────────────────────────────────────
print()
print(f"{'CLASS':<{ROW_W}} {'PRECISION':>12} {'RECALL':>10} {'F1':>10} {'SUPPORT':>10}")
print("-" * (ROW_W + 46))

total_tp = total_fp = total_fn = 0
macro_p = macro_r = macro_f1 = 0.0

for i, cls in enumerate(all_classes):
    tp = matrix[i][i]
    fp = sum(matrix[r][i] for r in range(n)) - tp   # col sum - diag
    fn = sum(matrix[i][c] for c in range(n)) - tp   # row sum - diag
    support = sum(matrix[i])

    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    macro_p  += prec
    macro_r  += recall
    macro_f1 += f1
    total_tp += tp
    total_fp += fp
    total_fn += fn

    print(f"{cls:<{ROW_W}} {prec:>12.4f} {recall:>10.4f} {f1:>10.4f} {support:>10,}")

print("-" * (ROW_W + 46))
macro_p  /= n
macro_r  /= n
macro_f1 /= n
accuracy = sum(matrix[i][i] for i in range(n)) / len(y_true)

micro_prec   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
micro_f1     = 2 * micro_prec * micro_recall / (micro_prec + micro_recall) if (micro_prec + micro_recall) else 0.0

print(f"{'MACRO AVG':<{ROW_W}} {macro_p:>12.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {len(y_true):>10,}")
print(f"{'MICRO AVG':<{ROW_W}} {micro_prec:>12.4f} {micro_recall:>10.4f} {micro_f1:>10.4f} {len(y_true):>10,}")
print(f"\nOverall accuracy: {accuracy:.4f}  ({sum(matrix[i][i] for i in range(n)):,} / {len(y_true):,})")


# ── 8. Save confusion matrix as YAML ──────────────────────────────────────────
cm_out = BASE / "output" / "confusion_matrix.yaml"
cm_data = {
    "classes": all_classes,
    "matrix": matrix,
    "metrics": {
        cls: {
            "precision": round(
                matrix[i][i] / sum(matrix[r][i] for r in range(n)), 4
            ) if sum(matrix[r][i] for r in range(n)) else 0.0,
            "recall": round(
                matrix[i][i] / sum(matrix[i]), 4
            ) if sum(matrix[i]) else 0.0,
            "support": sum(matrix[i]),
        }
        for i, cls in enumerate(all_classes)
    },
    "accuracy": round(accuracy, 4),
    "macro_f1": round(macro_f1, 4),
}
with cm_out.open("w", encoding="utf-8") as f:
    yaml.dump(cm_data, f, allow_unicode=True, sort_keys=False)
print(f"\nConfusion matrix saved to {cm_out}")
print("Done.")
