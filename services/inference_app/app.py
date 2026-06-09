"""
Gender Bias Detection Inference Web Application
Uses minilm_gender_bias_v2 model for text classification
"""

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model configuration
# Default points to the repo-level models/ folder; download weights from the
# Google Drive link in models/README.md first, or override with MODEL_PATH env var.
import os
from pathlib import Path

MODEL_NAME = "minilm_gender_bias_v2"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / MODEL_NAME
MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))

# Label configuration
LABELS = {0: "GB-ATTACK", 1: "GB-NORMATIVE", 2: "GB-SEX", 3: "neutral", 4: "meta_counter", 5: "gendered_insult"}

LABEL_DESCRIPTIONS = {
    "GB-ATTACK": "Gender-based attack or harassment",
    "GB-NORMATIVE": "Gender-based normative stereotypes",
    "GB-SEX": "Gender-based sexual harassment or inappropriate sexual content",
    "neutral": "Neutral content with no gender bias detected",
    "meta_counter": "Meta-commentary countering gender bias",
    "gendered_insult": "Gendered insults or derogatory language",
}

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None


def load_model():
    """Load the model and tokenizer"""
    global tokenizer, model, device

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info(f"Loading model from {MODEL_PATH}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")

        # Load tokenizer - use xlm-roberta-base as a compatible tokenizer
        logger.info("Loading tokenizer for model compatibility")
        try:
            # Try to load from the model directory first
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            logger.info("Tokenizer loaded from model directory")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {MODEL_PATH}: {e}")
            logger.info("Using compatible xlm-roberta-base tokenizer")

            # Use a compatible tokenizer from HuggingFace hub
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            logger.info("Tokenizer loaded from xlm-roberta-base")

        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict(text):
    """
    Perform inference on input text

    Args:
        text (str): Input text to classify

    Returns:
        dict: Prediction results with label and confidence scores
    """
    if not text or not text.strip():
        return {"error": "Input text cannot be empty"}

    try:
        # Tokenize input
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Get prediction results
        pred_id = torch.argmax(logits, dim=-1).item()
        pred_label = LABELS[pred_id]
        confidence = probabilities[0][pred_id].item()

        # Get all confidence scores
        all_scores = {}
        for label_id, label_name in LABELS.items():
            all_scores[label_name] = float(probabilities[0][label_id].item())

        return {
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "description": LABEL_DESCRIPTIONS.get(pred_label, ""),
            "all_scores": all_scores,
            "input_text": text,
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}


@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Text input is required"}), 400

        result = predict(text)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in API predict: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": str(device)})


if __name__ == "__main__":
    # Load model on startup
    load_model()

    # Run Flask app
    app.run(debug=True, host="0.0.0.0", port=6000)
