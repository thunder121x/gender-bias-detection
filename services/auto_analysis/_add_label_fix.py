import json
import uuid

def cell(cell_type, source):
    c = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

with open("services/auto_analysis/labeled_analysis.ipynb") as f:
    nb = json.load(f)

new_cells = [

cell("markdown", (
    "---\n"
    "## 4 — Fix Label Space (CRITICAL)\n\n"
    "Normalize to **7 canonical classes**:\n\n"
    "| Raw labels | Canonical |\n"
    "|---|---|\n"
    "| `GB-ATTACK` | `GB_ATTACK` |\n"
    "| `GB-NORMATIVE` | `GB_NORMATIVE` |\n"
    "| `GB-SEX` | `GB_SEX` |\n"
    "| `gendered_insult`, `GENDERED_INSULT` | `GENDERED_INSULT` |\n"
    "| `non-gb`, `NON-GB`, `NON-GB (D)` | `NON_GB` |\n"
    "| `meta_counter` | `META_COUNTER` |\n"
    "| `neutral` | `NEUTRAL` |\n"
)),

cell("code", (
    "LABEL_MAP = {\n"
    '    "GB-ATTACK":       "GB_ATTACK",\n'
    '    "GB_ATTACK":       "GB_ATTACK",\n'
    '    "GB-NORMATIVE":    "GB_NORMATIVE",\n'
    '    "GB_NORMATIVE":    "GB_NORMATIVE",\n'
    '    "GB-SEX":          "GB_SEX",\n'
    '    "GB_SEX":          "GB_SEX",\n'
    '    "GENDERED_INSULT": "GENDERED_INSULT",\n'
    '    "gendered_insult": "GENDERED_INSULT",\n'
    '    "NON-GB":          "NON_GB",\n'
    '    "NON-GB (D)":      "NON_GB",\n'
    '    "non-gb":          "NON_GB",\n'
    '    "NON_GB":          "NON_GB",\n'
    '    "meta_counter":    "META_COUNTER",\n'
    '    "META_COUNTER":    "META_COUNTER",\n'
    '    "neutral":         "NEUTRAL",\n'
    '    "NEUTRAL":         "NEUTRAL",\n'
    "}\n\n"
    'CANONICAL = ["GB_ATTACK", "GB_NORMATIVE", "GB_SEX",\n'
    '             "GENDERED_INSULT", "NON_GB", "META_COUNTER", "NEUTRAL"]\n\n'
    "def fix_label(lbl):\n"
    "    mapped = LABEL_MAP.get(str(lbl).strip())\n"
    "    if mapped is None:\n"
    "        print(f'  WARNING unmapped: {lbl!r}')\n"
    "    return mapped or lbl\n\n"
    'scraped_df["true_fixed"] = scraped_df["true_label"].map(fix_label)\n'
    'scraped_df["pred_fixed"] = scraped_df["predicted_label"].map(fix_label)\n\n'
    'print(f"Unique true labels:  {scraped_df[\'true_label\'].nunique()} -> {scraped_df[\'true_fixed\'].nunique()}")\n'
    'print(f"Unique pred labels:  {scraped_df[\'predicted_label\'].nunique()} -> {scraped_df[\'pred_fixed\'].nunique()}")\n\n'
    "mapping_check = (\n"
    '    scraped_df[["true_label", "true_fixed"]]\n'
    "    .drop_duplicates().sort_values('true_fixed')\n"
    "    .rename(columns={'true_label': 'raw', 'true_fixed': 'canonical'})\n"
    ")\n"
    "print('\\nLabel mapping:')\n"
    "display(mapping_check)\n"
)),

cell("markdown", "### 4.1  Distribution Before vs After Fix"),

cell("code", (
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n\n"
    "for ax, col, title, color in zip(\n"
    "    axes,\n"
    '    ["true_label", "true_fixed"],\n'
    '    ["Before fix (true labels)", "After fix (canonical 7 classes)"],\n'
    '    ["#aec7e8", "steelblue"]\n'
    "):\n"
    "    counts = scraped_df[col].value_counts().sort_values()\n"
    "    bars = ax.barh(counts.index, counts.values, color=color, edgecolor='white')\n"
    "    for bar, val in zip(bars, counts.values):\n"
    "        ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,\n"
    "                f'{val:,}', va='center', fontsize=9)\n"
    "    ax.set_title(title, fontsize=12)\n"
    "    ax.set_xlabel('Count')\n"
    "    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))\n"
    "    ax.spines[['top', 'right']].set_visible(False)\n\n"
    "plt.suptitle('True Label Distribution — Before vs After Fix', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig('output/fix_label_distribution.png', dpi=150)\n"
    "plt.show()\n"
)),

cell("markdown", "### 4.2  Confusion Matrix — Fixed Label Space"),

cell("code", (
    "from sklearn.metrics import confusion_matrix, classification_report\n\n"
    "cm_fix = confusion_matrix(\n"
    "    scraped_df['true_fixed'], scraped_df['pred_fixed'], labels=CANONICAL\n"
    ")\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(20, 7))\n\n"
    "sns.heatmap(cm_fix, annot=True, fmt='d', cmap='Blues',\n"
    "            xticklabels=CANONICAL, yticklabels=CANONICAL,\n"
    "            linewidths=0.4, linecolor='lightgrey',\n"
    "            ax=axes[0], cbar_kws={'shrink': 0.7})\n"
    "axes[0].set_title('Confusion Matrix — counts', fontsize=12)\n"
    "axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')\n"
    "axes[0].tick_params(axis='x', rotation=35)\n\n"
    "cm_fix_norm = cm_fix.astype(float) / cm_fix.sum(axis=1, keepdims=True).clip(min=1)\n"
    "sns.heatmap(cm_fix_norm, annot=True, fmt='.2f', cmap='YlOrRd',\n"
    "            xticklabels=CANONICAL, yticklabels=CANONICAL,\n"
    "            vmin=0, vmax=1, linewidths=0.4, linecolor='lightgrey',\n"
    "            ax=axes[1], cbar_kws={'shrink': 0.7})\n"
    "axes[1].set_title('Confusion Matrix — row-normalised (recall view)', fontsize=12)\n"
    "axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')\n"
    "axes[1].tick_params(axis='x', rotation=35)\n\n"
    "plt.suptitle('Fixed Label Space — Canonical 7 Classes', fontsize=13, y=1.01)\n"
    "plt.tight_layout()\n"
    "plt.savefig('output/fix_confusion_matrix.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved -> output/fix_confusion_matrix.png')\n"
)),

cell("markdown", "### 4.3  Per-class Metrics — After Fix"),

cell("code", (
    "report_fix = classification_report(\n"
    "    scraped_df['true_fixed'], scraped_df['pred_fixed'],\n"
    "    labels=CANONICAL, output_dict=True, zero_division=0\n"
    ")\n"
    "fix_df = pd.DataFrame(report_fix).T.loc[CANONICAL].copy()\n"
    "fix_df['support'] = fix_df['support'].astype(int)\n\n"
    "print('Per-class metrics AFTER label fix:\\n')\n"
    "display(fix_df.style\n"
    "    .format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}', 'support': '{:,.0f}'})\n"
    "    .background_gradient(subset=['f1-score'], cmap='RdYlGn', vmin=0, vmax=1)\n"
    "    .background_gradient(subset=['precision'], cmap='Blues',  vmin=0, vmax=1)\n"
    "    .background_gradient(subset=['recall'],    cmap='Greens', vmin=0, vmax=1)\n"
    "    .set_caption('Fixed label space — per-class metrics')\n"
    ")\n"
)),

cell("code", (
    "# Before vs After comparison table\n"
    "report_raw = classification_report(y_true, y_pred, output_dict=True, zero_division=0)\n\n"
    "rows = []\n"
    "for avg_name in ['macro avg', 'weighted avg']:\n"
    "    for metric in ['precision', 'recall', 'f1-score']:\n"
    "        rows.append({\n"
    "            'avg': avg_name, 'metric': metric,\n"
    "            'before_fix': report_raw[avg_name][metric],\n"
    "            'after_fix':  report_fix[avg_name][metric],\n"
    "            'delta': report_fix[avg_name][metric] - report_raw[avg_name][metric],\n"
    "        })\n\n"
    "compare_df = pd.DataFrame(rows)\n"
    "acc_before = (scraped_df['true_label'] == scraped_df['predicted_label']).mean()\n"
    "acc_after  = (scraped_df['true_fixed'] == scraped_df['pred_fixed']).mean()\n\n"
    "print(f'Accuracy before fix : {acc_before*100:.2f}%')\n"
    "print(f'Accuracy after  fix : {acc_after*100:.2f}%')\n"
    "print(f'Delta               : {(acc_after-acc_before)*100:+.2f}%\\n')\n\n"
    "display(compare_df.style\n"
    "    .format({'before_fix': '{:.4f}', 'after_fix': '{:.4f}', 'delta': '{:+.4f}'})\n"
    "    .background_gradient(subset=['delta'], cmap='RdYlGn', vmin=-0.1, vmax=0.3)\n"
    "    .set_caption('Before vs After label fix')\n"
    ")\n"
)),

cell("markdown", "### 4.4  F1 Score Lift — Before vs After Fix"),

cell("code", (
    "# Aggregate raw f1 into canonical buckets for a fair comparison\n"
    "raw_cls_report = classification_report(\n"
    "    y_true, y_pred, labels=sorted(set(y_true)|set(y_pred)),\n"
    "    output_dict=True, zero_division=0\n"
    ")\n"
    "merge_map = {\n"
    '    "GB_ATTACK":      ["GB-ATTACK"],\n'
    '    "GB_NORMATIVE":   ["GB-NORMATIVE"],\n'
    '    "GB_SEX":         ["GB-SEX"],\n'
    '    "GENDERED_INSULT":["GENDERED_INSULT", "gendered_insult"],\n'
    '    "NON_GB":         ["NON-GB", "NON-GB (D)", "non-gb"],\n'
    '    "META_COUNTER":   ["meta_counter"],\n'
    '    "NEUTRAL":        ["neutral"],\n'
    "}\n"
    "raw_canonical_f1 = {}\n"
    "for canon, raws in merge_map.items():\n"
    "    total_sup = sum(raw_cls_report.get(r, {}).get('support', 0) for r in raws)\n"
    "    if total_sup == 0:\n"
    "        raw_canonical_f1[canon] = 0.0; continue\n"
    "    raw_canonical_f1[canon] = sum(\n"
    "        raw_cls_report.get(r, {}).get('f1-score', 0) *\n"
    "        raw_cls_report.get(r, {}).get('support', 0) for r in raws\n"
    "    ) / total_sup\n\n"
    "fix_f1 = {cls: report_fix[cls]['f1-score'] for cls in CANONICAL}\n\n"
    "x = np.arange(len(CANONICAL)); w = 0.35\n"
    "fig, ax = plt.subplots(figsize=(13, 5))\n"
    "b1 = ax.bar(x - w/2, [raw_canonical_f1[c] for c in CANONICAL], w,\n"
    "            label='Before fix', color='#aec7e8', edgecolor='white')\n"
    "b2 = ax.bar(x + w/2, [fix_f1[c]            for c in CANONICAL], w,\n"
    "            label='After fix',  color='steelblue', edgecolor='white')\n"
    "for bar in list(b1) + list(b2):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n"
    "            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)\n"
    "ax.set_xticks(x); ax.set_xticklabels(CANONICAL, rotation=20, ha='right')\n"
    "ax.set_ylabel('F1-score'); ax.set_ylim(0, 1.05)\n"
    "ax.set_title('F1 Score — Before vs After Label Fix', fontsize=12)\n"
    "ax.legend(); ax.spines[['top', 'right']].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig('output/fix_f1_lift.png', dpi=150)\n"
    "plt.show()\n"
)),

cell("markdown", "### 4.5  Impact Summary"),

cell("code", (
    "acc_before = (scraped_df['true_label'] == scraped_df['predicted_label']).mean()\n"
    "acc_after  = (scraped_df['true_fixed'] == scraped_df['pred_fixed']).mean()\n"
    "mf1_before = report_raw['macro avg']['f1-score']\n"
    "mf1_after  = report_fix['macro avg']['f1-score']\n"
    "wf1_before = report_raw['weighted avg']['f1-score']\n"
    "wf1_after  = report_fix['weighted avg']['f1-score']\n\n"
    "n_classes_before = len(set(y_true) | set(y_pred))\n\n"
    "lines = [\n"
    "    'LABEL FIX IMPACT SUMMARY',\n"
    "    '-' * 52,\n"
    "    f\"{'':30} {'BEFORE':>10} {'AFTER':>10} {'DELTA':>10}\",\n"
    "    '-' * 52,\n"
    "    f\"  Accuracy             {acc_before*100:>10.2f}% {acc_after*100:>9.2f}%  {(acc_after-acc_before)*100:>+9.2f}%\",\n"
    "    f\"  Macro F1             {mf1_before:>10.4f}  {mf1_after:>9.4f}  {mf1_after-mf1_before:>+9.4f}\",\n"
    "    f\"  Weighted F1          {wf1_before:>10.4f}  {wf1_after:>9.4f}  {wf1_after-wf1_before:>+9.4f}\",\n"
    "    '-' * 52,\n"
    "    f\"  Classes before fix : {n_classes_before}   Classes after fix : {len(CANONICAL)}\",\n"
    "    '-' * 52,\n"
    "    'MERGES APPLIED',\n"
    "    '  NON-GB / NON-GB (D) / non-gb  ->  NON_GB',\n"
    "    '  GENDERED_INSULT / gendered_insult  ->  GENDERED_INSULT',\n"
    "    '  GB-ATTACK  ->  GB_ATTACK',\n"
    "    '  GB-NORMATIVE  ->  GB_NORMATIVE',\n"
    "    '  GB-SEX  ->  GB_SEX',\n"
    "    '  meta_counter  ->  META_COUNTER',\n"
    "    '  neutral  ->  NEUTRAL',\n"
    "]\n"
    "print('\\n'.join(lines))\n"
)),

]

nb["cells"].extend(new_cells)
with open("services/auto_analysis/labeled_analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done — total cells: {len(nb['cells'])}")
