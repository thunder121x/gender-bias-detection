#!/usr/bin/env python3
"""
Convert training data from tag-based format to JSON schema format
Old format: <GB-NORMATIVE>text</GB-NORMATIVE> <GB-ATTACK>text</GB-ATTACK>
New format: {"spans": [{"label": "GB-NORMATIVE", "text": "text"}, ...]}
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

def extract_spans_from_tags(text: str) -> List[Dict[str, str]]:
    """
    Extract spans from tagged text format.
    Supports: <GB-ATTACK>text</GB-ATTACK>, <GB-NORMATIVE>text</GB-NORMATIVE>, <GB-SEX>text</GB-SEX>
    Also handles cases with no tags (no bias found)
    """
    spans = []
    
    # Pattern to match XML tags with their content
    pattern = r'<(GB-ATTACK|GB-NORMATIVE|GB-SEX)>(.*?)</\1>'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        label = match.group(1)
        span_text = match.group(2).strip()
        
        if span_text:  # Only add non-empty spans
            spans.append({
                "label": label,
                "text": span_text
            })
    
    return spans

def convert_single_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single training example from tag format to JSON schema format.
    
    Input structure:
    {
        "text": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input_text}\n[/INST] {tagged_response}</s>"
    }
    
    Output structure:
    {
        "messages": [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{input_text}"},
            {"role": "assistant", "content": "{json_response}"}
        ]
    }
    """
    
    text = example.get('text', '')
    
    # Parse the Llama format
    # Extract system prompt
    sys_match = re.search(r'<<SYS>>\n(.*?)\n<</SYS>>', text, re.DOTALL)
    system_prompt = sys_match.group(1) if sys_match else ""
    
    # Extract user input (between </SYS>> and [/INST])
    user_match = re.search(r'<</SYS>>\n\n(.*?)\n\[/INST\]', text, re.DOTALL)
    user_input = user_match.group(1) if user_match else ""
    
    # Extract assistant response (after [/INST] and before </s>)
    response_match = re.search(r'\[/INST\]\s*(.*?)</s>$', text, re.DOTALL)
    response_text = response_match.group(1).strip() if response_match else ""
    
    # Extract spans from the response
    spans = extract_spans_from_tags(response_text)
    
    # Create JSON schema response
    assistant_response = {
        "spans": spans
    }
    
    # Build the new format
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_response, ensure_ascii=False)
            }
        ]
    }

def convert_file(input_path: str, output_path: str, max_examples: int = None) -> int:
    """
    Convert entire JSONL file from tag format to JSON schema format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        max_examples: Maximum number of examples to process (None = all)
    
    Returns:
        Number of examples processed
    """
    
    count = 0
    error_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if max_examples and count >= max_examples:
                break
            
            try:
                example = json.loads(line)
                converted = convert_single_example(example)
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                count += 1
                
                if count % 1000 == 0:
                    print(f"Processed {count} examples...", file=sys.stderr)
                    
            except Exception as e:
                error_count += 1
                print(f"Error at line {line_num}: {e}", file=sys.stderr)
                continue
    
    return count, error_count

def main():
    """Main conversion function"""
    
    # Define paths
    base_dir = Path(__file__).resolve().parents[3] / "services" / "lora_finetuning" / "training_data"
    
    # Convert training data
    print("Converting train.jsonl...", file=sys.stderr)
    train_count, train_errors = convert_file(
        str(base_dir / "train.jsonl"),
        str(base_dir / "train_new.jsonl")
    )
    print(f"Train: Converted {train_count} examples ({train_errors} errors)", file=sys.stderr)
    
    # Convert validation data
    print("\nConverting val.jsonl...", file=sys.stderr)
    val_count, val_errors = convert_file(
        str(base_dir / "val.jsonl"),
        str(base_dir / "val_new.jsonl")
    )
    print(f"Val: Converted {val_count} examples ({val_errors} errors)", file=sys.stderr)
    
    print(f"\n✓ Total: {train_count + val_count} examples converted", file=sys.stderr)
    print(f"✓ Output files: train_new.jsonl, val_new.jsonl", file=sys.stderr)

if __name__ == "__main__":
    main()
