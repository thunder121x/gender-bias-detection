#!/usr/bin/env python3
"""
Script for inference - detecting gender bias in text.

Usage:
    python scripts/03_inference.py --model-dir models/checkpoint-500 \
        --text "นี่คือข้อความทดสอบ" \
        --confidence 0.5
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import BiasDetector


def main():
    parser = argparse.ArgumentParser(
        description="Detect gender bias in text"
    )
    parser.add_argument(
        '--model-dir',
        required=True,
        help='Path to fine-tuned model directory'
    )
    parser.add_argument(
        '--text',
        help='Input text to analyze (if not provided, read from stdin)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for bias detection (0-1)'
    )
    parser.add_argument(
        '--highlight',
        action='store_true',
        help='Output highlighted text'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    # Validate model directory
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Initialize detector
    detector = BiasDetector(args.model_dir)
    
    # Get input text
    if args.text:
        text = args.text
    else:
        print("Enter text (press Ctrl+D when done):")
        text = sys.stdin.read()
    
    if not text.strip():
        print("Error: No input text provided")
        sys.exit(1)
    
    # Run detection
    result = detector.detect_bias(text, args.confidence)
    
    # Output results
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.highlight:
        highlighted = detector.highlight_text(result)
        print("\nHighlighted Text:")
        print(highlighted)
        print(f"\nSummary: {result['summary']['biased_count']}/{result['summary']['total_sentences']} sentences contain bias")
    else:
        # Print summary
        print(f"\nBias Detection Results:")
        print(f"Total sentences: {result['summary']['total_sentences']}")
        print(f"Biased sentences: {result['summary']['biased_count']}")
        print(f"Bias percentage: {result['summary']['bias_percentage']:.1f}%")
        
        if result['biased_sentences']:
            print(f"\nBiased Sentences:")
            for bias in result['biased_sentences']:
                print(f"  [{bias['index']}] ({bias['confidence']:.2%}) {bias['text']}")
                if bias['bias_spans']:
                    for span in bias['bias_spans']:
                        print(f"       -> {span['text']}")


if __name__ == '__main__':
    main()
