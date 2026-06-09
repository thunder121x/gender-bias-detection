# Auto-Analysis Error Module

Complete error analysis pipeline for gender bias detection validation.

## Overview

This module analyzes incorrect predictions from the auto-analysis service validation (105,114 records) and provides structured insights for improving the classification system.

**Key Metrics**:
- **Items Analyzed**: 18,975 incorrect predictions
- **Analysis Views**: 2 (by correct label, by predicted label)
- **Categories**: 8 label types across 2 organizational structures
- **Output**: Separated YAML files + improvement summaries

## Quick Start

```bash
# Navigate to module
cd services/auto_analysis/analysis/error_analysis

# Run analysis (uses previous incorrect_items.yaml)
python3 main.py

# View results
ls -la output/group_by_correct_label/
ls -la output/group_by_predicted_label/
```

**Results are automatically saved** to:
- `output/group_by_correct_label/` - Grouped by what should have been labeled
- `output/group_by_predicted_label/` - Grouped by what model predicted

## Module Structure

```
error_analysis/
├── __init__.py                    # Module initialization
├── main.py                        # CLI entry point (executable)
├── analyzer.py                    # Core analysis logic
├── utils.py                       # YAML I/O & formatting utilities
├── requirements.txt               # Dependencies (pyyaml)
├── QUICK_START.md                # Quick reference guide
├── SYNTHESIZER_V3_STRATEGY.md    # Implementation roadmap for improvements
└── output/                        # Generated analysis results
    ├── group_by_correct_label/   # 11 label types
    │   ├── neutral.yaml          # 13,062 items (69%)
    │   ├── meta_counter.yaml     # 3,060 items (16%)
    │   ├── SUMMARY.yaml          # Analysis summary
    │   └── [10 more label files]
    └── group_by_predicted_label/ # 6 predicted label types
        ├── GB-ATTACK.yaml        # 7,883 items (42%)
        ├── GB-NORMATIVE.yaml     # 5,627 items (30%)
        ├── SUMMARY.yaml          # Analysis summary
        └── [4 more label files]
```

## What Each File Does

### Python Modules

**main.py** (executable)
- CLI interface with optional `--input-file` and `--output-dir` arguments
- Orchestrates full analysis pipeline
- Reports progress and results

**analyzer.py**
- `ErrorAnalyzer` class: organizes items and generates insights
- Methods:
  - `analyze_by_correct_label()` - What each correct label got wrong
  - `analyze_by_predicted_label()` - What each prediction was about
  - Internal: pattern detection, improvement suggestions

**utils.py**
- `load_incorrect_items()` - Read YAML input
- `save_separated_items()` - Write organized YAML files
- `save_summary()` - Write analysis summaries
- `print_analysis_summary()` - Pretty-print results

### Documentation

**QUICK_START.md**
- What the module does
- How to run it
- Output structure explanation
- Key findings summary
- Integration points

**SYNTHESIZER_V3_STRATEGY.md**
- Detailed error analysis by category
- Root cause analysis for top 3 issues
- Implementation roadmap with priorities
- Specific prompt improvements needed
- Seed pool expansion guidelines
- Expected improvement metrics

## Key Findings

### Error Distribution

| Rank | Issue | Count | % | Root Cause |
|---|---|---|---|---|
| #1 | NEUTRAL → GB-ATTACK | 7,883 | 42% | Over-triggers on gender mention |
| #2 | NEUTRAL → GB-NORMATIVE | 5,627 | 30% | Confuses critique with norm enforcement |
| #3 | GENDERED_INSULT missing | 1,797 | 9% | Insufficient training examples |

These 3 issues account for **81% of all errors**.

### By Correct Label (Model Blindspots)

- **NEUTRAL** (13,062 errors): Model thinks gender mention = gender bias
- **meta_counter** (3,060 errors): Misses counter-speech and social critique
- **GB-NORMATIVE** (1,244 errors): Confuses stereotype discussion with enforcement
- **GB-ATTACK** (756 errors): Reverses with GB-NORMATIVE in 40% of cases
- **GENDERED_INSULT** (147 errors): Almost always missed or conflated with GB

## Using Results

### For Synthesizer v3 Improvements

See **SYNTHESIZER_V3_STRATEGY.md** for complete roadmap. Quick summary:

**Phase 1: Prompt Enhancements** (High Impact)
```
- GB-ATTACK: Add "must apply to ENTIRE gender group" criterion
- GB-NORMATIVE: Distinguish prescriptive vs. descriptive language
- gendered_insult: Emphasize "specific person, not group" distinction
```

**Phase 2: Seed Pool Expansion** (High Impact)
```
- non-gb-insult: 12 → 40 examples (3x for critical category)
- Others: 15 → 30 examples (2x for better coverage)
```

**Expected Improvement**: 75-80% error reduction (18,975 → 4,000-5,000 errors)

### For Dataset Augmentation

- Use separated files to identify missing categories
- `group_by_correct_label/gendered_insult.yaml` (147 real examples)
- Use as seed for synthetic data generation
- Focus on generating more GENDERED_INSULT and meta_counter examples

### For Guideline Clarification

- Review SUMMARY.yaml to understand annotation ambiguities
- Sample errors show real-world edge cases
- Use for training new annotators or refining guidelines

## Analysis Flow

```
auto_analysis/output/incorrect_items.yaml (18,975 items)
         ↓
    main.py
         ↓
┌────────┴────────┐
│                 │
analyzer.py    utils.py
│                 │
└────────┬────────┘
         ↓
    output/
    ├── group_by_correct_label/
    │   ├── neutral.yaml
    │   ├── ... (8 total)
    │   └── SUMMARY.yaml
    └── group_by_predicted_label/
        ├── GB-ATTACK.yaml
        ├── ... (5 total)
        └── SUMMARY.yaml
```

## Requirements

- Python 3.8+
- pyyaml

```bash
pip install -r requirements.txt
```

## CLI Usage

```bash
# Default (uses relative paths)
python3 main.py

# Custom paths
python3 main.py --input-file /full/path/incorrect_items.yaml --output-dir /full/path/output

# Help
python3 main.py --help
```

## Output Format

### YAML Files (grouped by label)

```yaml
records:
  - id: "Ugzguey9v1UUa1K95iF4AaABAg"
    text: "...(Thai text)..."
    predicted_label: "GB-ATTACK"
    correct_label: "neutral"
    reason: "...(Thai explanation)..."
```

### SUMMARY.yaml

```yaml
NEUTRAL:
  error_count: 13062
  most_common_issue: "Mispredicted as: GB-ATTACK"
  key_improvements:
    - "Reduce stereotyping detection - many items are social critique or humor"
    - "Improve attack vs criticism distinction - context matters"
    - "Better distinguish between insults and descriptive language"
    - "Improve gender norm detection - distinguish prescriptive vs descriptive"
    - "Include more context analysis in decision making"
```

## Integration with Other Services

### Upstream (Input)
- **auto_analysis service** (`../../../main.py`)
  - Validates 105,114 annotated records using Gemini API
  - Outputs: `../../output/incorrect_items.yaml`
  - Error analysis depends on this

### Downstream (Output)
- **synthesizer_v3** (`../../synthesizer_v3/`)
  - Uses error insights to improve training data generation
  - Reference: `SYNTHESIZER_V3_STRATEGY.md`
  - Implements: Better prompts and seed pools

## Examples

### Analyzing a Specific Category

```bash
# View all neutral items that were mispredicted as GB-ATTACK
cat output/group_by_correct_label/neutral.yaml | grep -A 10 "GB-ATTACK"

# Count how many each prediction type
wc -l output/group_by_correct_label/*.yaml
```

### Using Results for Training

```bash
# Extract improvement suggestions for synthesizer
cat output/group_by_correct_label/SUMMARY.yaml
cat output/group_by_predicted_label/SUMMARY.yaml

# Use real error examples as seed for synthetic generation
python3 generate_seed_examples.py < output/group_by_correct_label/gendered_insult.yaml
```

## Troubleshooting

| Issue | Solution |
|---|---|
| `incorrect_items.yaml` not found | Run auto_analysis first: `cd ../../../ && python3 main.py` |
| Permission denied | `chmod +x main.py` |
| ModuleNotFoundError | Run from error_analysis directory: `cd services/auto_analysis/analysis/error_analysis` |
| YAML parsing error | Check input file format matches expected schema |

## Performance

- **Processing time**: ~2-3 seconds for 18,975 items
- **Memory usage**: ~50-100 MB
- **Output size**: ~15-20 MB (YAML files)

## Future Enhancements

- [ ] Support for partial analysis (specific labels only)
- [ ] Statistical confidence intervals
- [ ] Confusion matrix visualization
- [ ] Trend analysis across multiple runs
- [ ] Integration with model retraining pipeline
- [ ] Web UI for browsing results

## Related Documentation

- **QUICK_START.md** - Quick reference and getting started
- **SYNTHESIZER_V3_STRATEGY.md** - Detailed improvement roadmap
- **annotation-guideline.md** - Label definitions (upstream reference)

## Contact & Feedback

For issues or improvements:
1. Review SYNTHESIZER_V3_STRATEGY.md
2. Check error_analysis/output/* for root causes
3. Implement suggested improvements in synthesizer_v3

---

**Created**: Apr 11, 2025
**Analysis Scope**: 18,975 incorrect predictions from 105,114 total annotations
**Purpose**: Improve gender bias detection through error-driven iteration
