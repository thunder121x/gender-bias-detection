"""
Inference pipeline for detecting and highlighting gender bias sentences.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification


class BiasDetector:
    """Detects gender bias at sentence level in paragraphs."""
    
    def __init__(self, model_dir: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the bias detector.
        
        Args:
            model_dir: Path to fine-tuned model directory
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        
        # Load label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        print(f"Bias detector initialized on device: {device}")
    
    def detect_bias(self, text: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect biased sentences in a paragraph.
        
        Args:
            text: Input paragraph text
            confidence_threshold: Confidence threshold for positive predictions
        
        Returns:
            {
                'paragraph': original text,
                'sentences': list of sentences,
                'biased_sentences': [
                    {
                        'text': sentence text,
                        'index': sentence index in paragraph,
                        'confidence': confidence score,
                        'tokens': list of tokens with predictions
                    }
                ],
                'summary': {
                    'total_sentences': int,
                    'biased_count': int,
                    'bias_percentage': float
                }
            }
        """
        # Split into sentences (simple space-based for Thai)
        sentences = text.split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        biased_sentences = []
        
        for sent_idx, sentence in enumerate(sentences):
            # Get predictions for sentence
            predictions = self._predict_sentence(sentence)
            
            if predictions['has_bias']:
                biased_sentences.append({
                    'text': sentence,
                    'index': sent_idx,
                    'confidence': predictions['confidence'],
                    'tokens': predictions['tokens'],
                    'bias_spans': predictions['bias_spans']
                })
        
        summary = {
            'total_sentences': len(sentences),
            'biased_count': len(biased_sentences),
            'bias_percentage': 100 * len(biased_sentences) / len(sentences) if sentences else 0
        }
        
        return {
            'paragraph': text,
            'sentences': sentences,
            'biased_sentences': biased_sentences,
            'summary': summary
        }
    
    def _predict_sentence(self, sentence: str) -> Dict:
        """
        Predict bias labels for a single sentence.
        
        Returns:
            {
                'has_bias': bool,
                'confidence': float,
                'tokens': [{'token': str, 'label': str, 'confidence': float}],
                'bias_spans': [{'text': str, 'start': int, 'end': int}]
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            sentence,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=2)[0]
        probabilities = torch.softmax(logits, dim=2)[0]
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # Process predictions
        token_predictions = []
        max_bias_conf = 0.0
        has_bias = False
        
        for token, pred, probs in zip(tokens, predictions, probabilities):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label_id = pred.item()
            label = self.id2label.get(label_id, 'O')
            confidence = probs[label_id].item()
            
            token_predictions.append({
                'token': token,
                'label': label,
                'confidence': float(confidence)
            })
            
            if label in ['B-BIAS', 'I-BIAS'] and confidence > max_bias_conf:
                max_bias_conf = confidence
                has_bias = True
        
        # Extract bias spans
        bias_spans = self._extract_bias_spans(token_predictions)
        
        return {
            'has_bias': has_bias,
            'confidence': float(max_bias_conf),
            'tokens': token_predictions,
            'bias_spans': bias_spans
        }
    
    def _extract_bias_spans(self, token_predictions: List[Dict]) -> List[Dict]:
        """
        Extract continuous bias spans from token predictions.
        
        Returns:
            List of bias spans with their text and positions
        """
        spans = []
        current_span = None
        
        for idx, pred in enumerate(token_predictions):
            if pred['label'] in ['B-BIAS', 'I-BIAS']:
                if current_span is None:
                    current_span = {
                        'tokens': [pred['token']],
                        'start': idx
                    }
                else:
                    current_span['tokens'].append(pred['token'])
            else:
                if current_span is not None:
                    current_span['end'] = idx - 1
                    spans.append(current_span)
                    current_span = None
        
        # Handle last span
        if current_span is not None:
            current_span['end'] = len(token_predictions) - 1
            spans.append(current_span)
        
        # Convert tokens back to text
        for span in spans:
            span['text'] = ' '.join(span['tokens']).replace(' ##', '')
            del span['tokens']
        
        return spans
    
    def batch_detect_bias(self, texts: List[str], 
                         confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect bias in multiple paragraphs.
        
        Args:
            texts: List of paragraph texts
            confidence_threshold: Confidence threshold
        
        Returns:
            List of detection results
        """
        results = []
        for text in texts:
            result = self.detect_bias(text, confidence_threshold)
            results.append(result)
        
        return results
    
    def highlight_text(self, detection_result: Dict) -> str:
        """
        Create highlighted version of text with bias marked.
        
        Returns:
            Markdown text with bold bias sentences
        """
        sentences = detection_result['sentences']
        biased_indices = {
            sent['index'] for sent in detection_result['biased_sentences']
        }
        
        highlighted_sentences = []
        for idx, sent in enumerate(sentences):
            if idx in biased_indices:
                # Find confidence for this sentence
                conf = next(
                    (s['confidence'] for s in detection_result['biased_sentences'] 
                     if s['index'] == idx),
                    0.0
                )
                highlighted_sentences.append(f"**{sent}** ({conf:.2%})")
            else:
                highlighted_sentences.append(sent)
        
        return " ".join(highlighted_sentences)
