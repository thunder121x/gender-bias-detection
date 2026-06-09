#!/usr/bin/env python3
"""
Convert dataset from JSON schema format to inline tag format
Before: Assistant response is JSON {"spans": [...]}
After: Assistant response has inline tags like "text with <GB-ATTACK>tag</GB-ATTACK>"
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

def spans_to_inline_tags(text: str, spans: List[Dict]) -> str:
    """Convert span annotations to inline tags."""
    if not spans:
        return text
    
    # Sort spans by position (reverse order to avoid index shifting)
    sorted_spans = sorted(spans, key=lambda x: x['text'], reverse=False)
    
    # Build position-based replacements
    result = text
    for span in sorted(spans, key=lambda x: text.find(x['text']), reverse=True):
        span_text = span['text']
        label = span['label']
        
        # Find and replace first occurrence
        pos = result.find(span_text)
        if pos != -1:
            result = result[:pos] + f"<{label}>{span_text}</{label}>" + result[pos + len(span_text):]
    
    return result

def convert_dataset(input_file: str, output_file: str):
    """Convert entire dataset from JSON format to inline tags."""
    count = 0
    errors = 0
    
    print(f"Converting {input_file} -> {output_file}")
    print(f"Format: JSON spans -> Inline tags")
    
    with open(input_file, 'r', encoding='utf-8') as inf, \
         open(output_file, 'w', encoding='utf-8') as outf:
        
        for line_num, line in enumerate(inf, 1):
            try:
                example = json.loads(line)
                messages = example.get("messages", [])
                
                # Convert last message (assistant) from JSON to inline tags
                if messages and messages[-1]['role'] == 'assistant':
                    try:
                        response = json.loads(messages[-1]['content'])
                        spans = response.get('spans', [])
                        user_text = messages[-2]['content']  # Get user input
                        
                        # Convert spans to inline tags in user text
                        tagged_text = spans_to_inline_tags(user_text, spans)
                        messages[-1]['content'] = tagged_text
                        
                    except (json.JSONDecodeError, IndexError, KeyError) as e:
                        print(f"⚠️  Line {line_num}: Could not parse assistant response: {e}")
                        errors += 1
                        continue
                
                # Write converted example
                outf.write(json.dumps(example, ensure_ascii=False) + '\n')
                count += 1
                
                if line_num % 1000 == 0:
                    print(f"  Converted {line_num} examples...")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: Invalid JSON: {e}")
                errors += 1
    
    print(f"✓ Converted {count} examples")
    if errors:
        print(f"⚠️  {errors} errors encountered")
    return count, errors

if __name__ == "__main__":
    # Convert train and val datasets
    train_input = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/train.jsonl"
    train_output = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/train_inline_tags.jsonl"
    
    val_input = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/val.jsonl"
    val_output = str(Path(__file__).resolve().parents[3] / "services/lora_finetuning/training_data") + "/val_inline_tags.jsonl"
    
    print("=" * 80)
    print("CONVERTING DATASET FORMAT: JSON -> INLINE TAGS")
    print("=" * 80)
    
    train_count, train_errors = convert_dataset(train_input, train_output)
    val_count, val_errors = convert_dataset(val_input, val_output)
    
    print("\n" + "=" * 80)
    print(f"Train: {train_count} examples ({train_errors} errors)")
    print(f"Val: {val_count} examples ({val_errors} errors)")
    print(f"\n✓ Output files:")
    print(f"  → {train_output}")
    print(f"  → {val_output}")
    print("=" * 80)
