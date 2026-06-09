#!/usr/bin/env python3
"""
Error Analysis Main Script

Analyzes incorrect predictions from the auto-analysis service.
Separates errors by correct_label and predicted_label, generating improvement summaries.

Usage:
    python3 main.py [--input-file path/to/incorrect_items.yaml] [--output-dir path/to/output]
"""

import argparse
import sys
from pathlib import Path

from analyzer import ErrorAnalyzer
from utils import load_incorrect_items, save_separated_items, save_summary, print_analysis_summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze errors from auto-analysis service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python3 main.py
  
  # Specify custom input and output paths
  python3 main.py --input-file ~/errors.yaml --output-dir ~/analysis_results
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default='../../output/incorrect_items.yaml',
        help='Path to incorrect_items.yaml (default: ../../output/incorrect_items.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_file = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Validate input file
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print(f"\nDid you run the auto-analysis service first?")
        print(f"Expected file: services/auto_analysis/output/incorrect_items.yaml")
        sys.exit(1)
    
    print(f"📂 Loading incorrect items from: {input_file}")
    
    try:
        incorrect_items = load_incorrect_items(str(input_file))
        print(f"✓ Loaded {len(incorrect_items)} incorrect items\n")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    
    if not incorrect_items:
        print("No incorrect items to analyze. Exiting.")
        sys.exit(0)
    
    # Initialize analyzer
    print("📊 Analyzing errors...\n")
    analyzer = ErrorAnalyzer(incorrect_items)
    
    # Analysis 1: By Correct Label
    print("=" * 70)
    print("GROUPING BY CORRECT_LABEL")
    print("=" * 70)
    
    print("\n📁 Saving separated items by correct_label...")
    items_by_correct = analyzer.get_all_items_by_correct_label()
    save_separated_items(str(output_dir), 'group_by_correct_label', items_by_correct)
    
    print("\n📝 Generating improvement summaries by correct_label...")
    analysis_correct = analyzer.analyze_by_correct_label()
    save_summary(str(output_dir), 'group_by_correct_label', analysis_correct)
    
    print_analysis_summary(analysis_correct, 'group_by_correct_label')
    
    # Analysis 2: By Predicted Label
    print("\n\n" + "=" * 70)
    print("GROUPING BY PREDICTED_LABEL")
    print("=" * 70)
    
    print("\n📁 Saving separated items by predicted_label...")
    items_by_predicted = analyzer.get_all_items_by_predicted_label()
    save_separated_items(str(output_dir), 'group_by_predicted_label', items_by_predicted)
    
    print("\n📝 Generating improvement summaries by predicted_label...")
    analysis_predicted = analyzer.analyze_by_predicted_label()
    save_summary(str(output_dir), 'group_by_predicted_label', analysis_predicted)
    
    print_analysis_summary(analysis_predicted, 'group_by_predicted_label')
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"\nGenerated directories:")
    print(f"  • {output_dir}/group_by_correct_label/")
    print(f"  • {output_dir}/group_by_predicted_label/")
    print(f"\nEach directory contains:")
    print(f"  • Individual YAML files for each label (e.g., neutral.yaml)")
    print(f"  • SUMMARY.yaml with improvement suggestions")


if __name__ == '__main__':
    main()
