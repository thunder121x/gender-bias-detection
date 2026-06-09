#!/usr/bin/env python
"""
Test script for gender bias detection model
Verifies the model loads and can make predictions
"""

import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Weights live in the repo-level models/ folder (see models/README.md for the
# Drive download link); override with MODEL_PATH env var.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "minilm_gender_bias_v2"
MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))

LABELS = {
    0: "GB-ATTACK",
    1: "GB-NORMATIVE",
    2: "GB-SEX",
    3: "neutral",
    4: "meta_counter",
    5: "gendered_insult"
}

def test_model():
    """Test model loading and inference"""
    
    print("=" * 60)
    print("Gender Bias Detection Model Test")
    print("=" * 60)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Load model first to get config
    print(f"\n🤖 Loading model from {MODEL_PATH}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
        print(f"  - Model type: {model.config.model_type}")
        print(f"  - Hidden size: {model.config.hidden_size}")
        print(f"  - Number of labels: {model.config.num_labels}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Load tokenizer
    print(f"\n📦 Loading tokenizer...")
    try:
        try:
            # Try to load from model directory first
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True
            )
            print(f"✓ Tokenizer loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"  (Could not load from {MODEL_PATH}: {type(e).__name__})")
            print(f"  Using compatible xlm-roberta-base tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            print("✓ Tokenizer loaded from xlm-roberta-base")
        
        print(f"  - Tokenizer vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return False
    
    # Test predictions
    test_texts = [
        "This is a neutral statement about the weather.",
        "Women are bad drivers.",
        "Girls are naturally bad at math.",
    ]
    
    print("\n" + "=" * 60)
    print("Running Test Predictions")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] Text: '{text}'")
        
        try:
            # Tokenize
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get results
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_label = LABELS[pred_id]
            confidence = probabilities[0][pred_id].item()
            
            print(f"  → Label: {pred_label}")
            print(f"  → Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # Show top 3 scores
            top_3 = torch.topk(probabilities[0], 3)
            print("  → Top scores:")
            for j, (score, idx) in enumerate(zip(top_3.values, top_3.indices), 1):
                print(f"     {j}. {LABELS[idx.item()]}: {score.item():.4f}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_model()
    exit(0 if success else 1)

