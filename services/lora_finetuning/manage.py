#!/usr/bin/env python3
"""
Thai Gender Bias Span Detector — LoRA Fine-tuning Service
Main entry point for all operations (data prep, training, inference)

Location: services/lora_finetuning/
"""

import sys
import os
from pathlib import Path

# Add current directory to path
SERVICE_DIR = Path(__file__).parent
sys.path.insert(0, str(SERVICE_DIR))

def print_menu():
    """Display main menu"""
    print("\n" + "="*80)
    print("THAI GENDER BIAS SPAN DETECTOR — LoRA FINE-TUNING SERVICE")
    print("="*80)
    print("\nAvailable Commands:\n")
    print("  1. validate    → Validate system prompt across files")
    print("  2. prepare     → Prepare training data from synthesized JSON")
    print("  3. train       → Start fine-tuning on RTX 6000 Blackwell")
    print("  4. inference   → Run inference (interactive or batch)")
    print("  5. help        → Show detailed help")
    print("  6. exit        → Exit\n")

def validate():
    """Run system prompt validation"""
    print("\n" + "="*80)
    print("VALIDATING SYSTEM PROMPT")
    print("="*80)
    os.system(f"cd {SERVICE_DIR} && python3 validate_system_prompt.py")

def prepare():
    """Run data preparation"""
    print("\n" + "="*80)
    print("PREPARING TRAINING DATA")
    print("="*80)
    os.system(f"cd {SERVICE_DIR} && python3 finetune_qwen_span_detector.py")

def train():
    """Start fine-tuning"""
    print("\n" + "="*80)
    print("STARTING FINE-TUNING (30-60 min on RTX 6000 Blackwell)")
    print("="*80)
    os.system(f"cd {SERVICE_DIR} && python3 finetune_qwen_lora.py")

def inference():
    """Run inference"""
    print("\nInference Mode:")
    print("  1. interactive → Single text detection")
    print("  2. batch       → Process JSONL file")
    print("  3. back        → Back to main menu\n")
    
    choice = input("Choose mode: ").strip().lower()
    
    if choice == "interactive" or choice == "1":
        os.system(f"cd {SERVICE_DIR} && python3 inference_qwen_span.py --mode interactive")
    elif choice == "batch" or choice == "2":
        input_file = input("Input JSONL file: ").strip()
        output_file = input("Output file: ").strip()
        os.system(f"cd {SERVICE_DIR} && python3 inference_qwen_span.py --mode batch --input {input_file} --output {output_file}")
    else:
        print("Back to main menu...")

def show_help():
    """Show detailed help"""
    print("\n" + "="*80)
    print("DETAILED HELP")
    print("="*80)
    help_text = """
    
QUICK START:
  1. Validate: python3 manage.py validate
  2. Prepare:  python3 manage.py prepare
  3. Train:    python3 manage.py train
  4. Infer:    python3 manage.py inference

DIRECTORY STRUCTURE:
  services/lora_finetuning/
  ├── finetune_qwen_span_detector.py    [Data Preparation]
  ├── finetune_qwen_lora.py             [Training Pipeline]
  ├── inference_qwen_span.py            [Inference Engine]
  ├── validate_system_prompt.py         [Validation]
  ├── training_data/                    [Generated Data]
  │   ├── train.jsonl (22.8k samples)
  │   └── val.jsonl (1.2k samples)
  ├── qwen_gb_detector_lora/            [Output Model]
  └── *.md                              [Documentation]

DOCUMENTATION:
  - FINETUNING_README.md       → Quick start guide
  - LORA_FINETUNING_GUIDE.md   → Detailed reference
  - IMPLEMENTATION_SUMMARY.md  → Implementation details
  - QUICK_REFERENCE.txt        → Quick reference card

KEY POINTS:
  - Model: Qwen 3.5 2B-Instruct
  - Method: LoRA 16-bit (NOT QLoRA)
  - Hardware: RTX 6000 Blackwell (96GB VRAM)
  - Data: 24k samples (22.8k train / 1.2k val)
  - Training Time: 30-60 minutes
  - Output Tags: <GB-ATTACK>, <GB-NORMATIVE>, <GB-SEX>

SYSTEM PROMPT:
  The system prompt is BYTE-IDENTICAL across:
  - Data preparation (training data generation)
  - Fine-tuning (training)
  - Inference (detection)
  
  This ensures perfect alignment between training and inference.

EXPECTED PERFORMANCE:
  - Accuracy: 96-98%
  - Precision (GB): 95%
  - Recall (GB): 93%
  - FPR (non-GB-meta): 2%

TROUBLESHOOTING:
  - CUDA not found:           → nvidia-smi
  - OOM (out of memory):      → Reduce batch_size in finetune_qwen_lora.py
  - Tags not recognized:      → Run validate first
  - High false-positive rate: → See LORA_FINETUNING_GUIDE.md

NEXT STEPS:
  1. Run: python3 manage.py validate
  2. Run: python3 manage.py prepare
  3. Run: python3 manage.py train
  4. Run: python3 manage.py inference
"""
    print(help_text)

def main():
    """Interactive menu"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║  THAI GENDER BIAS SPAN DETECTOR — LoRA FINE-TUNING SERVICE           ║")
    print("║  For RTX 6000 Blackwell (96GB VRAM)                                   ║")
    print("╚" + "="*78 + "╝")
    
    while True:
        print_menu()
        choice = input("Enter command (1-6): ").strip().lower()
        
        if choice in ["1", "validate"]:
            validate()
        elif choice in ["2", "prepare"]:
            prepare()
        elif choice in ["3", "train"]:
            train()
        elif choice in ["4", "inference"]:
            inference()
        elif choice in ["5", "help"]:
            show_help()
        elif choice in ["6", "exit", "quit", "q"]:
            print("\nGoodbye! 👋\n")
            break
        else:
            print("❌ Invalid command. Please try again.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode
        cmd = sys.argv[1].lower()
        
        if cmd == "validate":
            validate()
        elif cmd == "prepare":
            prepare()
        elif cmd == "train":
            train()
        elif cmd == "inference":
            inference()
        elif cmd == "help":
            show_help()
        else:
            print(f"Unknown command: {cmd}")
            print("\nAvailable commands: validate, prepare, train, inference, help")
            sys.exit(1)
    else:
        # Interactive mode
        main()
