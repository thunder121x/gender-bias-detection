#!/usr/bin/env python3
"""
Convert dataset from JSON schema format to inline tag format
Takes the Input: text from user message and converts JSON spans to inline tags
"""

import json
import re
from pathlib import Path
from typing import List, Dict

def spans_to_inline_tags(text: str, spans: List[Dict]) -> str:
    """Convert span annotations to inline tags."""
    if not spans:
        return text
    
    # Sort spans by position in text (reverse for safe replacement)
    sorted_spans = sorted(spans, key=lambda x: text.find(x.get('text', '')), reverse=True)
    
    result = text
    for span in sorted_spans:
        span_text = span.get('text', '')
        label = span.get('label', '')
        
        if not span_text or not label:
            continue
        
        # Find and replace - using careful approach to handle multiple occurrences
        pos = result.find(span_text)
        if pos != -1:
            result = result[:pos] + f"<{label}>{span_text}</{label}>" + result[pos + len(span_text):]
    
    return result

def extract_input_text(user_message: str) -> str:
    """Extract the 'Input: ...' part from user message."""
    # Look for "Input: " pattern
    match = re.search(r'Input:\s*(.+)$', user_message, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no "Input:" prefix, return whole message (shouldn't happen)
    return user_message

def convert_dataset(input_file: str, output_file: str):
    """Convert entire dataset from JSON format to inline tags."""
    count = 0
    errors = 0
    
    print(f"Converting {input_file}")
    print(f"Format: Extract Input -> JSON spans -> Inline tags")
    
    with open(input_file, 'r', encoding='utf-8') as inf, \
         open(output_file, 'w', encoding='utf-8') as outf:
        
        for line_num, line in enumerate(inf, 1):
            try:
                example = json.loads(line)
                messages = example.get("messages", [])
                
                if len(messages) < 3:
                    errors += 1
                    continue
                
                # Extract input text from user message
                user_msg = messages[1]['content']
                input_text = extract_input_text(user_msg)
                
                # Parse assistant JSON response
                assistant_response = messages[2]['content']
                response_data = json.loads(assistant_response)
                spans = response_data.get('spans', [])
                
                # Convert spans to inline tags
                tagged_text = spans_to_inline_tags(input_text, spans)
                
                # Update assistant message with tagged text
                messages[2]['content'] = tagged_text
                
                # Write converted example
                outf.write(json.dumps(example, ensure_ascii=False) + '\n')
                count += 1
                
                if line_num % 1000 == 0:
                    print(f"  ✓ {line_num} examples...")
                    
            except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
                errors += 1
                if errors <= 5:  # Show first 5 errors only
                    print(f"  ⚠️  Line {line_num}: {type(e).__name__}: {str(e)[:100]}")
    
    print(f"✓ Converted {count} examples")
    if errors:
        print(f"⚠️  {errors} errors skipped")
    return count, errors

if __name__ == "__main__":
    # Convert train and val datasets
    train_input = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/train.jsonl"
    train_output = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/train_inline_tags_v2.jsonl"
    
    val_input = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/val.jsonl"
    val_output = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/val_inline_tags_v2.jsonl"
    
    print("=" * 80)
    print("CONVERTING DATASET: JSON -> INLINE TAGS (v2)")
    print("=" * 80)
    
    train_count, train_errors = convert_dataset(train_input, train_output)
    val_count, val_errors = convert_dataset(val_input, val_output)
    
    print("\n" + "=" * 80)
    print(f"✓ Train: {train_count} examples ({train_errors} errors)")
    print(f"✓ Val: {val_count} examples ({val_errors} errors)")
    print(f"\nOutput files:")
    print(f"  → {train_output}")
    print(f"  → {val_output}")
    print("=" * 80)
