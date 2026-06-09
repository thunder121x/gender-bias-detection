# 🚀 Auto-Analysis Service - START HERE

Welcome! This service automatically validates your annotated gender bias data using Google's Gemini API.

## What Does It Do?

- ✓ Reads 105,114 annotated records from `services/auto_annalysis/assets/scraped_data.yaml`
- ✓ Validates each label using Gemini API (10 concurrent requests)
- ✓ Compares against Gender Bias annotation guidelines
- ✓ Returns **only incorrect items** in same YAML format
- ✓ Generates accuracy metrics

## Quick Start (3 Steps)

### 1️⃣ Install Dependencies
```bash
cd services/auto_analysis
pip install -r requirements.txt
```

### 2️⃣ Verify Setup
```bash
python3 test.py
```
Expected output: `✓ All tests passed!`

### 3️⃣ Run Full Analysis
```bash
./run.sh
```

Results saved to `output/` directory

## Output

After running, you get:

1. **incorrect_items.yaml** - Items with wrong labels
2. **summary.yaml** - Accuracy metrics and stats

## What's in This Directory?

| File | Purpose |
|------|---------|
| `main.py` | Main service entry point |
| `config.py` | Configuration settings |
| `gemini_validator.py` | Gemini API integration |
| `run.sh` | Convenient runner script |
| `test.py` | Test configuration |
| `demo.py` | See it in action (first 2 batches) |
| `QUICKSTART.md` | 30-second setup |
| `USAGE.md` | Complete guide |
| `OUTPUT_FORMAT.md` | Understanding results |
| `requirements.txt` | Dependencies |

## Try the Demo First

Want to see it in action with just 2 batches?

```bash
python3 demo.py
```

This shows you exactly what the validation output looks like.

## Understanding Your Results

### The Good News 📊
```
Total Incorrect: 3482 / 105114
Accuracy: 96.68%
```

### What You Get 📁
1. **incorrect_items.yaml** - Review these
2. **summary.yaml** - See the stats

### Next Steps 🎯
1. Review incorrect items
2. Identify patterns
3. Fix your model
4. Re-run validation

## Configuration

Default settings (fast, cost-effective):
- Batch size: 100 items
- Concurrent: 10 requests
- Model: gemini-3.1-flash-lite-preview

Edit `config.py` to customize.

## Troubleshooting

**Tests fail?**
```bash
python3 test.py
```
This tells you exactly what's wrong.

**Need help?**
- See `USAGE.md` for detailed guide
- See `OUTPUT_FORMAT.md` for result examples
- Check `QUICKSTART.md` for basic setup

## Processing Timeline

- 📊 Total records: 105,114
- 📦 Batches: 1,052 (100 items each)
- 🚀 Speed: 10 concurrent requests
- ⏱️ Estimated time: 30-60 minutes

## Is This Right For Me?

Use this service if you:
- ✓ Have annotated data with predicted labels
- ✓ Want to validate accuracy automatically
- ✓ Need to identify mislabeled items
- ✓ Have a Gemini API key

## Get Started Now

```bash
# One command to rule them all:
./run.sh

# Monitor progress in real-time
tail -f output/summary.yaml  # In another terminal
```

## Questions?

Check these in order:
1. `QUICKSTART.md` - Basic setup
2. `USAGE.md` - Complete guide
3. `OUTPUT_FORMAT.md` - Understanding results
4. `test.py` - Diagnose issues

---

**Ready?** Run: `./run.sh`
