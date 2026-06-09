#!/usr/bin/env python3
"""
Utility script to inspect and validate training data
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List


def analyze_dataset(file_path: str, sample_size: int = 5):
    """Analyze and print dataset statistics."""
    
    print(f"\n{'=' * 80}")
    print(f"Dataset Analysis: {file_path}")
    print(f"{'=' * 80}\n")
    
    examples = []
    spans_per_example = []
    label_counts = Counter()
    avg_text_length = 0
    avg_spans = 0
    
    # Load data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line)
                examples.append(example)
                
                # Extract spans
                assistant_content = example["messages"][2]["content"]
                response = json.loads(assistant_content)
                spans = response.get("spans", [])
                
                spans_per_example.append(len(spans))
                
                for span in spans:
                    label_counts[span["label"]] += 1
                    avg_text_length += len(span["text"])
                
                avg_spans += len(spans)
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"⚠️ Error reading line: {e}")
    
    # Calculate statistics
    total_examples = len(examples)
    total_spans = sum(spans_per_example)
    avg_spans = avg_spans / total_examples if total_examples > 0 else 0
    avg_text_length = avg_text_length / total_spans if total_spans > 0 else 0
    
    # Print statistics
    print(f"📊 STATISTICS:")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Total spans: {total_spans:,}")
    print(f"  Avg spans per example: {avg_spans:.2f}")
    print(f"  Examples with bias: {sum(1 for x in spans_per_example if x > 0):,}")
    print(f"  Examples without bias: {sum(1 for x in spans_per_example if x == 0):,}")
    print(f"  Avg span text length: {avg_text_length:.1f} chars")
    
    print(f"\n📈 LABEL DISTRIBUTION:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_spans * 100) if total_spans > 0 else 0
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🔀 SPAN COUNT DISTRIBUTION:")
    span_counts = Counter(spans_per_example)
    for count in sorted(span_counts.keys()):
        examples_count = span_counts[count]
        percentage = (examples_count / total_examples * 100)
        print(f"  {count} span(s): {examples_count:,} examples ({percentage:.1f}%)")
    
    # Show samples
    print(f"\n📝 SAMPLE EXAMPLES (first {sample_size}):")
    print(f"{'-' * 80}")
    
    for i, example in enumerate(examples[:sample_size]):
        print(f"\nExample {i+1}:")
        
        # User input
        user_content = example["messages"][1]["content"]
        input_text = user_content.split("Input: ")[-1] if "Input: " in user_content else user_content
        print(f"  Input: {input_text[:100]}...")
        
        # Spans
        assistant_content = example["messages"][2]["content"]
        response = json.loads(assistant_content)
        spans = response.get("spans", [])
        
        if spans:
            print(f"  Spans found: {len(spans)}")
            for span in spans:
                print(f"    - {span['label']}: '{span['text'][:60]}...'")
        else:
            print(f"  No bias detected")
    
    print(f"\n{'=' * 80}\n")


def validate_format(file_path: str, validate_all: bool = False) -> bool:
    """Validate dataset format."""
    
    print(f"\n{'=' * 80}")
    print(f"Format Validation: {file_path}")
    print(f"{'=' * 80}\n")
    
    errors = []
    valid_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line)
                
                # Check structure
                if "messages" not in example:
                    errors.append(f"Line {line_num}: Missing 'messages' key")
                    continue
                
                messages = example["messages"]
                
                # Check message count
                if len(messages) != 3:
                    errors.append(f"Line {line_num}: Expected 3 messages, got {len(messages)}")
                    continue
                
                # Check message roles
                roles = [m.get("role") for m in messages]
                if roles != ["system", "user", "assistant"]:
                    errors.append(f"Line {line_num}: Invalid roles {roles}")
                    continue
                
                # Check assistant content is valid JSON
                assistant_content = messages[2].get("content", "")
                try:
                    response = json.loads(assistant_content)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON in assistant response: {e}")
                    continue
                
                # Check spans structure
                if "spans" not in response:
                    errors.append(f"Line {line_num}: Missing 'spans' in response")
                    continue
                
                spans = response["spans"]
                if not isinstance(spans, list):
                    errors.append(f"Line {line_num}: 'spans' is not a list")
                    continue
                
                # Check each span
                for span_idx, span in enumerate(spans):
                    if not isinstance(span, dict):
                        errors.append(f"Line {line_num}, span {span_idx}: Not a dict")
                        continue
                    
                    if "label" not in span or "text" not in span:
                        errors.append(f"Line {line_num}, span {span_idx}: Missing 'label' or 'text'")
                        continue
                    
                    label = span["label"]
                    if label not in ["GB-ATTACK", "GB-NORMATIVE", "GB-SEX"]:
                        errors.append(f"Line {line_num}, span {span_idx}: Invalid label '{label}'")
                
                valid_count += 1
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON: {e}")
            
            if not validate_all and len(errors) > 10:
                errors.append("... (too many errors, stopping validation)")
                break
    
    # Print results
    print(f"✓ Valid examples: {valid_count:,}")
    
    if errors:
        print(f"✗ Errors found: {len(errors)}")
        print(f"\nErrors:")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False
    else:
        print(f"✅ No errors found!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Inspect and validate training data")
    parser.add_argument("--file", type=str, required=True, help="JSONL file to inspect")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to show")
    parser.add_argument("--validate", action="store_true", help="Validate format")
    parser.add_argument("--validate-all", action="store_true", help="Validate entire file")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        return
    
    # Analyze
    analyze_dataset(args.file, args.samples)
    
    # Validate
    if args.validate or args.validate_all:
        validate_format(args.file, args.validate_all)


if __name__ == "__main__":
    main()
