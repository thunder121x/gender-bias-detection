# Quick Start

## 30 Second Setup

```bash
cd services/auto_analysis
pip install -r requirements.txt
python3 test.py    # Verify everything works
```

## Run Analysis

```bash
./run.sh    # Runs full validation (takes 30-60 minutes)
```

## Check Results

```bash
# View incorrect items
cat output/incorrect_items.yaml

# View summary
cat output/summary.yaml
```

## Demo (2 batches only)

```bash
python3 demo.py
```

## What It Does

1. ✓ Loads 105,114 annotated records from `services/auto_annalysis/assets/scraped_data.yaml`
2. ✓ Validates each label against annotation guidelines using Gemini API
3. ✓ Processes 100 items per batch with 10 concurrent requests
4. ✓ Returns only **incorrect items** in YAML format
5. ✓ Generates summary with accuracy metrics

## Output

- **incorrect_items.yaml** - Only mislabeled items (you review these)
- **summary.yaml** - Processing stats and accuracy %

## That's it! See USAGE.md for detailed information.
