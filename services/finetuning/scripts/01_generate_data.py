#!/usr/bin/env python3
"""
Script to generate synthetic training data for token classification model.

Usage:
    python scripts/01_generate_data.py --config config/config.yaml --num-samples 10000
"""

import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_augmenter import DataAugmenter


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data for gender bias detection")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of synthetic paragraphs to generate")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve input_dir relative to project root
    input_dir = config["data"]["input_dir"]
    if not os.path.isabs(input_dir):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up: scripts -> finetuning -> services -> project_root
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        input_dir = os.path.join(project_root, input_dir)
        print(f"Resolved path: {input_dir}")

    # Initialize augmenter
    augmenter = DataAugmenter(config)

    # Load sentences
    augmenter.load_sentences(input_dir)

    # Generate dataset
    dataset = augmenter.generate_dataset(num_samples=args.num_samples)

    # Save dataset
    output_dir = config["data"]["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    train_path, val_path, test_path = augmenter.save_dataset(dataset, output_dir)

    print("\nData generation complete!")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Test: {test_path}")


if __name__ == "__main__":
    main()
