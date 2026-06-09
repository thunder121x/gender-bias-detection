#!/usr/bin/env python3
"""
Main entry point for gender bias detection fine-tuning service.

Usage:
    python -m services.finetuning generate --num-samples 10000
    python -m services.finetuning train
    python -m services.finetuning infer --text "your text here"
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Gender Bias Token Classification Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate synthetic data:
    python -m services.finetuning generate --num-samples 10000
  
  Train model:
    python -m services.finetuning train
  
  Run inference:
    python -m services.finetuning infer --text "ผู้หญิงสมัยนี้นอกจากสวยแล้วมีอะไรอีก?"
  
  Get help for specific command:
    python -m services.finetuning generate --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic training data')
    gen_parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to generate')
    gen_parser.add_argument('--config', default='services/finetuning/config/config.yaml', help='Config file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', default='services/finetuning/config/config.yaml', help='Config file')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on text')
    infer_parser.add_argument('--model-dir', default='services/finetuning/models', help='Model directory')
    infer_parser.add_argument('--text', help='Input text')
    infer_parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    infer_parser.add_argument('--highlight', action='store_true', help='Highlight output')
    infer_parser.add_argument('--json', action='store_true', help='JSON output')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        from scripts.generate_data import main as generate_main
        sys.argv = ['generate_data.py', '--config', args.config, '--num-samples', str(args.num_samples)]
        generate_main()
    
    elif args.command == 'train':
        from scripts.train import main as train_main
        sys.argv = ['train.py', '--config', args.config]
        train_main()
    
    elif args.command == 'infer':
        from scripts.inference import main as infer_main
        sys.argv = ['inference.py', '--model-dir', args.model_dir]
        if args.text:
            sys.argv.extend(['--text', args.text])
        if args.highlight:
            sys.argv.append('--highlight')
        if args.json:
            sys.argv.append('--json')
        sys.argv.extend(['--confidence', str(args.confidence)])
        infer_main()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
