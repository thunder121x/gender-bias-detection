#!/usr/bin/env python3
"""
Script to generate training data in instruction-input-output format for LLM fine-tuning.
Converts sentences with gender bias spans into instruction data format.

Format:
{
  "instruction": "Identify and tag social bias in the following Thai text using <GB-NORM> for generalized bias and <GB-SEX> for sexual harassment.",
  "input": "ผู้หญิงทุกคนก็โง่ สวัสดี หีอย่างอูม",
  "output": "<GB-NORM>ผู้หญิงทุกคนก็โง่</GB-NORM>สวัสดี<GB-SEX>หีอย่างอูม</GB-SEX>"
}

Usage:
    python scripts/02_generate_instruction_data.py --num-samples 10000 --output-dir data/
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Gender bias tag mapping
BIAS_TAGS = {
    "GB-NORMATIVE": "<GB-NORMATIVE>",
    "GB-ATTACK": "<GB-ATTACK>",
    "GB-SEX": "<GB-SEX>",
}

INSTRUCTION = "Identify and tag social bias in the following Thai text using <GB-ATTACK> for attacks/derogatory terms, <GB-NORMATIVE> for stereotypes/gender roles, and <GB-SEX> for sexual harassment/objectification."


class InstructionDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.bias_sentences: Dict[str, List[str]] = {
            "GB-NORMATIVE": [],
            "GB-ATTACK": [],
            "GB-SEX": [],
        }
        self.neutral_sentences: List[str] = []

    def load_data(self, data_dir: str = "services/finetuning/data"):
        """Load existing training data and extract sentences."""
        print(f"Loading data from {data_dir}...")

        # Load train.jsonl to extract sentences with bias information
        train_file = os.path.join(data_dir, "train.jsonl")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Cannot find {train_file}")

        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Extract sentences and their bias labels
                sentences = data.get("sentences", [])
                bias_info = data.get("bias_info", [])
                sentence_labels = data.get("sentence_labels", [])

                for idx, sentence in enumerate(sentences):
                    if sentence_labels[idx] == 1:  # Biased sentence
                        # Find which bias type this sentence has
                        for bias in bias_info:
                            if bias["index"] == idx:
                                bias_type = bias["subtype"]
                                if bias_type in self.bias_sentences:
                                    self.bias_sentences[bias_type].append({
                                        "text": sentence,
                                        "bias_type": bias_type,
                                        "full_info": bias,
                                    })
                                break
                    else:  # Neutral sentence
                        self.neutral_sentences.append(sentence)

        print(f"Loaded:")
        for bias_type, sentences in self.bias_sentences.items():
            print(f"  {bias_type}: {len(sentences)} sentences")
        print(f"  Neutral: {len(self.neutral_sentences)} sentences")

    def create_tagged_sentence(self, sentence: str, bias_type: str) -> str:
        """
        Add tags to a biased sentence.
        
        For simplicity, tag the entire sentence or specific keywords.
        In a production system, this would use precise span information from bias_info.
        """
        if not bias_type or bias_type not in BIAS_TAGS:
            return sentence
        
        tag = BIAS_TAGS[bias_type]
        close_tag = tag.replace(">", ">").replace("<", "</")
        
        # For now, wrap the entire sentence with the tag
        # In a more sophisticated approach, would identify specific spans
        return f"{tag}{sentence}{close_tag}"

    def create_mixed_text(self, max_sentences: int = 5) -> Tuple[str, str]:
        """
        Create a text with mixed sentences (some biased, some not).
        Returns (input_text, output_text_with_tags)
        """
        num_sentences = random.randint(2, max_sentences)
        input_parts = []
        output_parts = []
        
        # Decide how many biased sentences to include (0-2 per text)
        num_biased = random.randint(0, min(2, num_sentences))
        biased_positions = set(random.sample(range(num_sentences), num_biased))
        
        for i in range(num_sentences):
            if i in biased_positions and num_biased > 0:
                # Add a biased sentence
                bias_type = random.choice(list(self.bias_sentences.keys()))
                if self.bias_sentences[bias_type]:
                    sentence = random.choice(self.bias_sentences[bias_type])["text"]
                    input_parts.append(sentence)
                    output_parts.append(self.create_tagged_sentence(sentence, bias_type))
                else:
                    # Fallback to neutral
                    if self.neutral_sentences:
                        sentence = random.choice(self.neutral_sentences)
                        input_parts.append(sentence)
                        output_parts.append(sentence)
            else:
                # Add a neutral sentence
                if self.neutral_sentences:
                    sentence = random.choice(self.neutral_sentences)
                    input_parts.append(sentence)
                    output_parts.append(sentence)
        
        input_text = " ".join(input_parts)
        # Join with space to separate sentences in output as well
        output_text = " ".join(output_parts)
        
        return input_text, output_text

    def generate_dataset(self, num_samples: int = 10000) -> List[Dict]:
        """Generate instruction-input-output format dataset."""
        print(f"\nGenerating {num_samples} samples...")
        
        dataset = []
        for i in range(num_samples):
            input_text, output_text = self.create_mixed_text()
            
            if input_text and output_text:
                sample = {
                    "instruction": INSTRUCTION,
                    "input": input_text,
                    "output": output_text,
                }
                dataset.append(sample)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        print(f"Generated {len(dataset)} valid samples")
        return dataset

    def save_dataset(self, dataset: List[Dict], output_dir: str) -> str:
        """Save dataset to JSONL format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "instruction_data.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"\nDataset saved to {output_file}")
        print(f"Total samples: {len(dataset)}")
        
        # Show example
        print("\nExample sample:")
        if dataset:
            example = dataset[0]
            print(f"  Instruction: {example['instruction'][:80]}...")
            print(f"  Input: {example['input'][:100]}...")
            print(f"  Output: {example['output'][:100]}...")
        
        return str(output_file)

    def split_dataset(self, dataset: List[Dict], train_ratio: float = 0.95) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and validation sets."""
        random.shuffle(dataset)
        split_idx = int(len(dataset) * train_ratio)
        return dataset[:split_idx], dataset[split_idx:]

    def save_split_datasets(self, dataset: List[Dict], output_dir: str, train_ratio: float = 0.95) -> Tuple[str, str]:
        """Save train and validation datasets separately."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_data, val_data = self.split_dataset(dataset, train_ratio)
        
        train_file = output_dir / "instruction_train.jsonl"
        val_file = output_dir / "instruction_val.jsonl"
        
        with open(train_file, "w", encoding="utf-8") as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        with open(val_file, "w", encoding="utf-8") as f:
            for sample in val_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"\nDatasets saved:")
        print(f"  Train: {train_file} ({len(train_data)} samples)")
        print(f"  Val: {val_file} ({len(val_data)} samples)")
        
        return str(train_file), str(val_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate instruction-input-output format training data for gender bias detection"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        default="services/finetuning/data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val files",
    )

    args = parser.parse_args()

    # Create generator
    generator = InstructionDataGenerator(seed=args.seed)

    # Load existing data
    try:
        generator.load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure train.jsonl exists in the output directory")
        return

    # Generate dataset
    dataset = generator.generate_dataset(num_samples=args.num_samples)

    if not dataset:
        print("Error: No samples generated")
        return

    # Save dataset
    if args.split:
        generator.save_split_datasets(dataset, args.output_dir)
    else:
        generator.save_dataset(dataset, args.output_dir)

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
