# Error Analysis Module - Quick Start

## What This Module Does

Analyzes **18,975 incorrect predictions** from the auto-analysis validation service and separates them into two organizational views:

1. **By Correct Label** (What should have been labeled)
2. **By Predicted Label** (What the model predicted instead)

Each view generates:
- Individual YAML files grouped by category
- SUMMARY.yaml with improvement suggestions
- Human-readable analysis reports

## Running the Analysis

### Basic Usage
```bash
cd services/auto_analysis/analysis/error_analysis
python3 main.py
```

### Custom Input/Output
```bash
python3 main.py \
  --input-file /path/to/incorrect_items.yaml \
  --output-dir /path/to/output
```

## Output Structure

```
output/
├── group_by_correct_label/
│   ├── neutral.yaml              # 13,062 items (69% of errors)
│   ├── meta_counter.yaml         # 3,060 items (16%)
│   ├── GB-NORMATIVE.yaml         # 1,244 items (7%)
│   ├── GB-ATTACK.yaml            # 756 items (4%)
│   ├── GB-SEX.yaml               # 442 items (2%)
│   ├── GENDERED_INSULT.yaml      # 147 items (1%)
│   ├── NON-GB.yaml               # 223 items (1%)
│   └── SUMMARY.yaml              # Analysis summary
│
└── group_by_predicted_label/
    ├── GB-ATTACK.yaml            # 7,883 predicted items
    ├── GB-NORMATIVE.yaml         # 5,627 predicted items
    ├── GENDERED_INSULT.yaml      # 1,797 predicted items
    ├── GB-SEX.yaml               # 1,582 predicted items
    ├── NEUTRAL.yaml              # 1,573 predicted items
    ├── meta_counter.yaml         # 513 predicted items
    └── SUMMARY.yaml              # Analysis summary
```

## Key Findings

### Top 3 Issues (75% of all errors)

| Issue | Count | % | Impact |
|---|---|---|---|
| NEUTRAL classified as GB-ATTACK | 7,883 | 42% | Over-triggers on gender mention |
| NEUTRAL classified as GB-NORMATIVE | 5,627 | 30% | Confuses critique with norm enforcement |
| GENDERED_INSULT missing | 1,797 | 9% | Under-represents personal insults |

### By Category

**NEUTRAL (should be)** → 13,062 errors
- 68% mispredicted as GB-ATTACK
- Root cause: Over-generalizing from gender mention
- Solution: Better context sensitivity

**GB-NORMATIVE (should be)** → 1,244 errors  
- 54% mispredicted as GB-ATTACK
- Root cause: Misidentifying stereotyping as attack
- Solution: Distinguish prescriptive vs. descriptive

**meta_counter (should be)** → 3,060 errors
- 60% mispredicted as GB-ATTACK
- Root cause: Not recognizing counter-speech
- Solution: Expand meta_counter examples

## Using Results for Training

1. **For synthesizer_v3 improvements**:
   - See: `SYNTHESIZER_V3_STRATEGY.md`
   - Use error examples to refine prompts and seed pools

2. **For dataset augmentation**:
   - Separated files show gaps in training data
   - gendered_insult.yaml shows 147 real examples
   - Use as seed for synthetic data generation

3. **For error analysis**:
   - SUMMARY.yaml in each folder lists key improvements
   - Sample errors show common patterns
   - Use for guideline clarification

## File Descriptions

### Python Modules

- **main.py**: CLI entry point, orchestrates analysis
- **analyzer.py**: Core ErrorAnalyzer class
  - Organizes items by label
  - Generates improvement suggestions
  - Analyzes error patterns
- **utils.py**: Utility functions
  - YAML I/O
  - File organization
  - Pretty printing

### Generated Files

- **group_by_correct_label/SUMMARY.yaml**: What each correct label type got wrong
- **group_by_predicted_label/SUMMARY.yaml**: Why model made each prediction
- **Individual label files**: Full YAML exports for deeper analysis

## Analysis Insights

### Group by Correct Label (What SHOULD Be)

Shows model blindspots by correct category:
- **neutral.yaml**: 13K items - model over-classifies as GB
- **meta_counter.yaml**: 3K items - model misses counter-speech
- **gendered_insult.yaml**: 147 items - model conflates with GB

**Use case**: Understand model limitations for each label type

### Group by Predicted Label (What Model Predicted)

Shows most common confusions:
- **GB-ATTACK.yaml**: 7,883 mispredictions - what triggered false positives?
- **GB-NORMATIVE.yaml**: 5,627 mispredictions - what looked like norms?
- **GENDERED_INSULT.yaml**: 1,797 mispredictions - what got classified wrong?

**Use case**: Debug model predictions, identify systematic errors

## Integration with Auto-Analysis Service

1. Run auto-analysis: `python3 ../../../main.py`
   - Generates: `../../output/incorrect_items.yaml` (18,975 items)

2. Run error analysis: `python3 main.py`
   - Consumes: auto-analysis output
   - Produces: organized analysis + summaries

3. Use results: Feed back to synthesizer_v3
   - Improve prompts based on error patterns
   - Expand seed pools in weak categories
   - Create better training data

## Requirements

- Python 3.8+
- pyyaml

Install:
```bash
pip install -r requirements.txt
```

## Troubleshooting

**Input file not found**:
```
❌ Error: Input file not found: /path/to/incorrect_items.yaml
Did you run the auto-analysis service first?
```
Solution: Run `python3 ../../../main.py` in auto_analysis directory first

**Permission denied**:
```bash
chmod +x main.py
```

**ModuleNotFoundError**:
```bash
cd services/auto_analysis/analysis/error_analysis
python3 main.py
```
(Must run from within error_analysis directory)

## Next Steps

1. **Review SUMMARY.yaml files** in both output directories
2. **Check SYNTHESIZER_V3_STRATEGY.md** for implementation roadmap
3. **Use individual label files** to understand specific error patterns
4. **Extract examples** for prompt improvements in synthesizer_v3

---

**Analysis Date**: Apr 11, 2025
**Items Analyzed**: 18,975 incorrect predictions
**Source**: Auto-analysis service validation output
**Next**: Feed insights into synthesizer_v3 training data generation
