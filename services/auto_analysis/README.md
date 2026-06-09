# Auto-Analysis Service

This service validates annotated data against the Gender Bias annotation guidelines using Google's Gemini API.

## Features

- **Batch Processing**: Processes 100 items per batch from the scraped data
- **Concurrent Processing**: Handles up to 10 concurrent API requests
- **Selective Output**: Returns only incorrect items in the same YAML format
- **Guideline-Based Validation**: Uses the annotation guideline to validate labels
- **Comprehensive Summary**: Generates processing summary and accuracy metrics

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The service automatically loads the API key from the `.env` file (already configured):

```bash
# .env file contains:
GEMINI_API_KEY=your-api-key
```

Or set via environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
```

The service reads from:
- `services/auto_analysis/assets/scraped_data.yaml` - Input data with predicted labels
- `services/auto_analysis/assets/prompt/annotation/annotation-guideline.md` - Annotation guidelines

## Usage

Run the service:

```bash
python main.py
```

## Output

The service generates:

1. **incorrect_items.yaml** - Only items with incorrect labels in the same format as input
2. **summary.yaml** - Processing summary with accuracy metrics

### Example Output Format

```yaml
records:
  - id: 'UgxTzXyCxlnww6zFwCR4AaABAg'
    text: 'Example text here'
    predicted_label: 'incorrect_label'
    correct_label: 'GB-ATTACK'
    reason: 'Text contains direct personal attack based on gender...'
```

## How It Works

1. **Loads** all records from `scraped_data.yaml`
2. **Chunks** them into batches of 100 items
3. **Validates** each batch using Gemini API with concurrent requests (max 10)
4. **Compares** predicted labels against annotation guidelines
5. **Collects** only incorrect items
6. **Saves** results in YAML format for review

## Processing Details

- Total records: ~105,114
- Batch size: 100 items
- Concurrent requests: 10 (max)
- Total batches: ~1,051
- Output: Only incorrect items + summary report

## Valid Labels

- `neutral` - No gender bias detected
- `GB-ATTACK` - Direct attack on gender/sexual orientation
- `GB-NORMATIVE` - Gender stereotype or role enforcement
- `GB-SEX` - Sexualized attack or body-based gender insult
- `meta_counter` - Meta commentary or counter-argument

## Error Handling

- Timeouts: Items timeout after 120 seconds per batch
- Retries: Failed batches are logged in summary
- Partial Results: Even if some batches fail, processed results are saved
- Fast Model: Uses gemini-3.1-flash-lite-preview for faster processing
