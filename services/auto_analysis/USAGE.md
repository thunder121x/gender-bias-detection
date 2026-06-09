# Auto-Analysis Service - Complete Usage Guide

## Overview

This service automatically validates annotated gender bias data against annotation guidelines using Google's Gemini API. It processes 105,114 records in batches of 100 items with up to 10 concurrent API requests.

### Key Features

✅ **Auto-Save**: Results are saved after every batch
✅ **Resume Capability**: Can resume from where it left off if interrupted
✅ **Progress Tracking**: Maintains progress.json with detailed state information
✅ **Incremental Output**: incorrect_items.yaml updates as processing continues

## Quick Start

### 1. Install Dependencies

```bash
cd services/auto_analysis
pip install -r requirements.txt
```

### 2. Configure API Key

Option A: Using .env file (already configured)
```bash
# The .env file already contains your API key
# Just run:
./run.sh
```

Option B: Using environment variable
```bash
export GEMINI_API_KEY="your-api-key-here"
python3 main.py
```

### 3. Run the Service

```bash
# Using the convenience script
./run.sh

# Or directly with Python
python3 main.py
```

## File Structure

```
auto_analysis/
├── main.py                 # Main service entry point (with auto-save & resume)
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── gemini_validator.py    # Gemini API integration
├── test.py               # Configuration tests
├── demo.py               # Demo script showing how it works
├── run.sh                # Convenience runner script
├── requirements.txt      # Python dependencies
├── .env                  # API key configuration
├── README.md            # Quick reference
└── output/              # Generated outputs
    ├── incorrect_items.yaml   # Items with incorrect labels (updated every batch)
    ├── summary.yaml           # Processing summary (updated every batch)
    └── progress.json          # Processing state (for resuming)
```

## How It Works

### Processing Flow

1. **Load Data**
   - Reads 105,114 records from `services/auto_analysis/assets/scraped_data.yaml`
   - Each record has: id, text, and predicted_label

2. **Batch Processing**
   - Splits records into batches of 100 items
   - Creates 1,052 total batches

3. **Concurrent Validation**
   - Sends up to 10 batches concurrently to Gemini API
   - Each request validates the annotation against guidelines
   - Gemini determines if labels are correct or incorrect

4. **Auto-Save & Progress Tracking**
   - After each batch, results are automatically saved
   - progress.json tracks which batch was last processed
   - incorrect_items.yaml is updated with new incorrect items
   - summary.yaml shows current progress

5. **Resume Capability**
   - If processing is interrupted (Ctrl+C, crash, timeout)
   - Run the script again and it will ask to resume
   - Picks up from the last successfully processed batch
   - No duplicates in output (uses tracking file)

### Annotation Guidelines Reference

The validation uses annotation guidelines with these valid labels:
- `neutral` - No gender bias detected
- `GB-ATTACK` - Direct attack based on gender/SOGI
- `GB-NORMATIVE` - Stereotype, gender role, norm enforcement, or policing
- `GB-SEX` - Sexualized attack or body-based gender insult
- `meta_counter` - Meta commentary or counter-argument

## Auto-Save and Resume Features

### What Gets Saved Automatically

After **every batch**:
1. ✅ Progress state (batch number, counts, timestamps)
2. ✅ Incorrect items found so far
3. ✅ Summary statistics with current progress

### How to Resume

```bash
# If processing was interrupted:
python3 main.py

# You'll see:
# Found existing progress file.
# Last processed batch: 250
# Incorrect items found so far: 1234
#
# Resume from last batch? (y/n, default=y): y
```

Then it will continue from batch 251 onwards.

### Progress Files

The service creates these files in `output/`:

**progress.json** - Contains:
```json
{
  "started_at": "2025-04-11T...",
  "last_updated": "2025-04-11T...",
  "last_batch": 250,
  "total_incorrect": 1234,
  "processed_batches": 250,
  "failed_batches": 2,
  "total_batches": 1052
}
```

**summary.yaml** - Shows current status:
```yaml
timestamp: '2025-04-11T...'
started_at: '2025-04-11T...'
total_records: 105114
total_incorrect: 1234
accuracy: '98.83%'
batches_processed: 250
batches_failed: 2
total_batches: 1052
is_complete: false
status: 'IN_PROGRESS'
```

**incorrect_items.yaml** - Updated incrementally:
```yaml
records:
  - id: '...'
    text: '...'
    predicted_label: '...'
    correct_label: '...'
    reason: '...'
  # ... more items
```

## Testing

### Run All Tests

```bash
python3 test.py
```

This validates:
- ✓ Configuration and environment
- ✓ Data files exist and are readable
- ✓ Gemini API connectivity
- ✓ Data loading and batching

### Run Demo

```bash
python3 demo.py
```

This runs a demo with just the first 2 batches to show you what validation output looks like.

## Configuration Options

Edit `config.py` to customize:

```python
BATCH_SIZE = 100              # Items per batch
MAX_CONCURRENT_REQUESTS = 10  # Max concurrent API calls
TIMEOUT_SECONDS = 120         # Timeout per batch
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
```

## Common Scenarios

### Scenario 1: Normal Full Run
```bash
python3 main.py
# Runs all 1,052 batches without interruption
# Saves results along the way
# Total time: 30-60 minutes
```

### Scenario 2: Interrupted (Ctrl+C)
```bash
python3 main.py
# [Processing... Ctrl+C after 250 batches]
# ⚠️ Processing interrupted by user!
# 📊 Progress saved:
#    Processed: 250 batches
#    Incorrect items found: 1234
# 🔄 To resume: Run the script again and it will continue from batch 251

# Later, resume:
python3 main.py
# Resume from last batch? (y/n, default=y): y
# Continues from batch 251
```

### Scenario 3: Network Timeout or Error
```bash
python3 main.py
# [Processing... network timeout after 500 batches]
# ❌ Error during processing: timeout
# 💾 Progress saved before error:
#    Processed: 500 batches
#    Incorrect items: 5000
# 🔄 To resume: Run the script again

# Resume:
python3 main.py
# Resume from last batch? (y/n, default=y): y
# Continues from batch 501
```

### Scenario 4: Monitor Progress Without Running
```bash
# While processing is running in another terminal:
cat output/progress.json
cat output/summary.yaml

# See real-time progress:
watch -n 5 'cat output/summary.yaml'
```

## Error Handling

- **Timeouts**: Batches timeout after 120 seconds per batch
- **Failed Batches**: Logged in progress and summary files
- **Partial Results**: Even if some batches fail, processed results are saved
- **Interruption**: Ctrl+C or crashes save progress before exiting

## Output Interpretation

### incorrect_items.yaml

Only items with **incorrect** labels:
- Fields: id, text, predicted_label, correct_label, reason
- Updates after each batch with new incorrect items
- Total size typically 1,000-5,000 items (1-5% of 105,114)

### summary.yaml

Current processing status:
- `status`: "IN_PROGRESS" or "COMPLETE"
- `is_complete`: true/false
- `accuracy`: Current accuracy %
- `batches_processed`: How many batches done
- `total_incorrect`: Count of incorrect items found

### progress.json

For resuming and tracking:
- `last_batch`: Last successfully processed batch number
- `total_incorrect`: Count so far
- `processed_batches`: How many batches completed
- `started_at`: When processing started

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Tests fail | Run `python3 test.py` to diagnose |
| API key error | Check `.env` has GEMINI_API_KEY |
| Data not found | Verify file paths in config.py |
| Slow processing | Check network, reduce batch size if needed |
| Resume not working | Check `output/progress.json` exists |
| Duplicate items in output | Should not happen; progress.json prevents this |

## Next Steps After Completion

1. Review `output/incorrect_items.yaml` for errors
2. Analyze patterns in mistakes
3. Identify model weaknesses
4. Retrain or fine-tune model
5. Re-run validation to verify improvements

## Support

For issues or questions:
- Check configuration with: `python3 test.py`
- Run demo: `python3 demo.py`
- Review annotation guidelines
- Check progress files: `cat output/progress.json`

```bash
python3 test.py
```

This validates:
- ✓ Configuration and environment
- ✓ Data files exist and are readable
- ✓ Gemini API connectivity
- ✓ Data loading and batching

### Run Demo

```bash
python3 demo.py
```

This runs a demo with just the first 2 batches to show you what validation output looks like.

## Output Format

### incorrect_items.yaml

```yaml
records:
  - id: 'UgxTzXyCxlnww6zFwCR4AaABAg'
    text: 'Example text that was misclassified'
    predicted_label: 'neutral'  # What the model predicted
    correct_label: 'GB-ATTACK'  # What it should be
    reason: 'Text contains direct derogatory language attacking gender...'
  
  - id: 'another_id'
    text: 'Another example...'
    predicted_label: 'neutral'
    correct_label: 'GB-NORMATIVE'
    reason: 'Reflects gender stereotype...'
```

### summary.yaml

```yaml
timestamp: '2025-04-11T10:30:45.123456'
total_records: 105114
total_incorrect: 1234  # Only incorrect items found
accuracy: '98.83%'     # Percentage of correct items
batch_count: 1052
successful_batches: 1050
failed_batches: 2
```

## Configuration Options

Edit `config.py` to customize:

```python
BATCH_SIZE = 100              # Items per batch
MAX_CONCURRENT_REQUESTS = 10  # Max concurrent API calls
TIMEOUT_SECONDS = 60          # Timeout per batch
GEMINI_MODEL = "gemini-2.5-flash"  # Model version
```

## Performance

- **Speed**: ~1052 batches, 10 concurrent = ~105 batch processing cycles
- **Cost**: Uses Gemini 2.5 Flash (cost-effective)
- **Processing Time**: Estimated 30-60 minutes for full dataset
- **Output Size**: Only incorrect items (typically 1-5% of total)

## Troubleshooting

### API Key Error
```
❌ GEMINI_API_KEY not found in environment variables
```
**Solution**: Set the API key in `.env` or environment variable

### Data Not Found
```
❌ Scraped data file not found
```
**Solution**: Ensure `services/auto_annalysis/assets/scraped_data.yaml` exists

### Timeout Error
```
Timeout after 60s
```
**Solution**: Increase `TIMEOUT_SECONDS` in config.py

### API Rate Limit
If you hit API rate limits, reduce `MAX_CONCURRENT_REQUESTS` in config.py

## Example Commands

```bash
# Run full validation
./run.sh

# Run with custom API key
GEMINI_API_KEY="your-key" python3 main.py

# Run demo to test
python3 demo.py

# Run tests
python3 test.py

# Check if a specific batch succeeded
tail -50 output/summary.yaml
```

## Understanding the Output

Only items with **incorrect labels** appear in the output. This is intentional:

- ✓ Correctly labeled items → Not included in output
- ✗ Incorrectly labeled items → Included in `incorrect_items.yaml`

The summary shows:
- **Total Records**: All 105,114 items processed
- **Total Incorrect**: Count of mislabeled items found
- **Accuracy**: Percentage of correctly labeled items
- **Batch Statistics**: Processing success rate

## Next Steps

1. Review `output/incorrect_items.yaml` for mislabeled items
2. Analyze patterns in mistakes
3. Retrain model with corrections if needed
4. Re-run validation after fixes

## Support

For issues or questions:
- Check configuration with: `python3 test.py`
- Run demo: `python3 demo.py`
- Review annotation guidelines: `services/auto_annalysis/assets/prompt/annotation/annotation-guideline.md`
