# Gender Bias Detection - Inference Website

A web-based text classification system for detecting gender bias in text using the **MiniLM Gender Bias v2** model.

## Features

- **Real-time Text Classification**: Classify text into 6 gender bias categories
- **Confidence Scores**: View detailed confidence scores for all categories
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Fast Inference**: Powered by MiniLM transformer for quick predictions
- **Interactive UI**: Modern, user-friendly interface with visual feedback

## Classification Categories

| Category | Description |
|----------|-------------|
| **GB-ATTACK** | Gender-based attack or harassment |
| **GB-NORMATIVE** | Gender-based normative stereotypes |
| **GB-SEX** | Gender-based sexual harassment or inappropriate sexual content |
| **neutral** | Neutral content with no gender bias detected |
| **meta_counter** | Meta-commentary countering gender bias |
| **gendered_insult** | Gendered insults or derogatory language |

## System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for faster inference)
- CUDA-compatible GPU (optional, but recommended for faster inference)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure the following files exist in the `minilm_gender_bias_v2` directory:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

## Usage

### Running the Application

Start the Flask development server:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Enter or paste text in the text area
3. Click "Analyze Text" or press `Ctrl+Enter` (Cmd+Enter on Mac)
4. View results including:
   - Primary classification label
   - Confidence score
   - Detailed category scores
   - Description of the classification

## Project Structure

```
gender-bias-detection/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── minilm_gender_bias_v2/         # Model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── templates/
│   └── index.html                 # HTML template
└── static/
    ├── style.css                  # Styling
    └── script.js                  # Frontend JavaScript
```

## API Endpoints

### POST /api/predict

Performs text classification.

**Request:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "predicted_label": "neutral",
  "confidence": 0.9876,
  "description": "Neutral content with no gender bias detected",
  "all_scores": {
    "GB-ATTACK": 0.0012,
    "GB-NORMATIVE": 0.0034,
    "GB-SEX": 0.0018,
    "neutral": 0.9876,
    "meta_counter": 0.0045,
    "gendered_insult": 0.0015
  },
  "input_text": "Your text here"
}
```

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "minilm_gender_bias_v2",
  "device": "cuda"
}
```

## Model Information

- **Model Name**: minilm_gender_bias_v2
- **Base Architecture**: BertForSequenceClassification
- **Tokenizer**: XLMRobertaTokenizer
- **Hidden Size**: 384
- **Number of Layers**: 12
- **Attention Heads**: 12
- **Max Sequence Length**: 512
- **Number of Labels**: 6

## Performance

- **Inference Time**: ~100-500ms per text (CPU), ~20-50ms (GPU)
- **Model Size**: ~134MB
- **Supported Languages**: Multilingual (especially optimized for Thai and English)

## Customization

### Changing Model Path

Edit the `MODEL_PATH` variable in `app.py`:

```python
MODEL_PATH = "./minilm_gender_bias_v2"  # Change this path
```

### Modifying Labels

Update the `LABELS` and `LABEL_DESCRIPTIONS` dictionaries in `app.py`:

```python
LABELS = {
    0: "YOUR_LABEL",
    # ...
}

LABEL_DESCRIPTIONS = {
    "YOUR_LABEL": "Description here",
    # ...
}
```

### Adjusting UI Colors

Edit the CSS variables in `static/style.css`:

```css
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    /* ... */
}
```

## Troubleshooting

### Model Loading Error
- Ensure all model files are present in `minilm_gender_bias_v2` directory
- Check file permissions
- Verify sufficient disk space

### CUDA/GPU Issues
- Install CUDA and cuDNN if you have an NVIDIA GPU
- If issues persist, CPU inference is still available (but slower)

### Port Already in Use
```bash
# Use a different port
python app.py  # Then modify app.run(port=5001)
```

### Memory Issues
- Reduce batch size in `predict()` function
- Clear browser cache
- Restart the application

## Advanced Usage

### Running in Production

For production deployment, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

Create a `.env` file for configuration:

```bash
FLASK_ENV=production
FLASK_DEBUG=0
MODEL_PATH=./minilm_gender_bias_v2
SERVER_PORT=5000
```

## License

This project is part of the gender-bias-detection research initiative.

## Support

For issues or questions, please refer to the project documentation or contact the development team.

---

**Last Updated**: April 2026
**Model Version**: minilm_gender_bias_v2

## Authors

- **Nanphat Tongsirisukool** — nanphatx@hotmail.com
- **Natcha Trairattanasak** — pnbookclub@gmail.com
