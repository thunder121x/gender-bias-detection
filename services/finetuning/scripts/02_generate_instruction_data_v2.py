#!/usr/bin/env python3
"""
Enhanced Data Generator for Thai Gender Bias Span Detection.
Generates 30,000 diverse samples covering different case scenarios:
- Case 1: Pure Bias (ALL BIAS) — 6,000-7,500 samples
- Case 2: Partial Bias (MIXED) — 15,000-16,500 samples
- Case 3: No Bias (ALL NEUTRAL) — 6,000-7,500 samples
- Case 4: Long Text + Multiple Biases — 3,000-4,500 samples
- Case 5: Edge Cases & Tricky Negatives — 1,500-3,000 samples

Usage:
    python scripts/02_generate_instruction_data_v2.py --num-samples 30000 --output-dir data/ --split
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
from enum import Enum

# Gender bias tag mapping
BIAS_TAGS = {
    "GB-NORMATIVE": "<GB-NORMATIVE>",
    "GB-ATTACK": "<GB-ATTACK>",
    "GB-SEX": "<GB-SEX>",
}

INSTRUCTION = "Identify and tag social bias in the following Thai text using <GB-ATTACK> for attacks/derogatory terms, <GB-NORMATIVE> for stereotypes/gender roles, and <GB-SEX> for sexual harassment/objectification."


class CaseType(Enum):
    """Different case scenarios for training data"""
    PURE_BIAS = "pure_bias"           # All/almost all sentences biased
    PARTIAL_BIAS = "partial_bias"     # Mix of biased and neutral
    NO_BIAS = "no_bias"               # All neutral, no bias
    LONG_MULTI = "long_multi"         # Long text (8-15 sent) with multiple biases
    EDGE_CASES = "edge_cases"         # Meta-commentary, counter-speech


class EnhancedInstructionDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.bias_sentences: Dict[str, List[Dict]] = {
            "GB-NORMATIVE": [],
            "GB-ATTACK": [],
            "GB-SEX": [],
        }
        self.neutral_sentences: List[str] = []
        self.meta_sentences: List[str] = []  # Counter-speech, meta-commentary

    def load_data(self, data_dir: str = "services/finetuning/data"):
        """Load existing training data and extract sentences."""
        print(f"Loading data from {data_dir}...")

        train_file = os.path.join(data_dir, "train.jsonl")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Cannot find {train_file}")

        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                sentences = data.get("sentences", [])
                bias_info = data.get("bias_info", [])
                sentence_labels = data.get("sentence_labels", [])

                for idx, sentence in enumerate(sentences):
                    if sentence_labels[idx] == 1:  # Biased sentence
                        for bias in bias_info:
                            if bias["index"] == idx:
                                bias_type = bias["subtype"]
                                if bias_type in self.bias_sentences:
                                    self.bias_sentences[bias_type].append({
                                        "text": sentence,
                                        "bias_type": bias_type,
                                    })
                                break
                    else:  # Neutral sentence
                        self.neutral_sentences.append(sentence)

        # Separate meta-commentary sentences (heuristic)
        neutral_with_keywords = [s for s in self.neutral_sentences 
                                 if any(kw in s for kw in ["ยังมี", "เห็น", "คน", "พูด", "น่า", "แนวคิด", "สังคม"])]
        self.meta_sentences = neutral_with_keywords[:len(neutral_with_keywords)//3]

        print(f"Loaded:")
        for bias_type, sentences in self.bias_sentences.items():
            print(f"  {bias_type}: {len(sentences)} sentences")
        print(f"  Neutral: {len(self.neutral_sentences)} sentences")
        print(f"  Meta-commentary candidates: {len(self.meta_sentences)} sentences")

    def create_tagged_sentence(self, sentence: str, bias_type: str) -> str:
        """Add tags to a biased sentence."""
        if not bias_type or bias_type not in BIAS_TAGS:
            return sentence
        
        tag = BIAS_TAGS[bias_type]
        close_tag = tag.replace("<", "</")
        
        return f"{tag}{sentence}{close_tag}"

    def case_pure_bias(self) -> Tuple[str, str]:
        """
        Case 1: PURE BIAS - Entire text is biased
        All or almost all sentences are tagged
        """
        num_sentences = random.randint(3, 6)  # Shorter but all biased
        output_parts = []
        
        for i in range(num_sentences):
            bias_type = random.choice(list(self.bias_sentences.keys()))
            if self.bias_sentences[bias_type]:
                sentence = random.choice(self.bias_sentences[bias_type])["text"]
                output_parts.append(self.create_tagged_sentence(sentence, bias_type))
        
        input_text = " ".join(s.replace(f"<{tag}>", "").replace(f"</{tag}>", "") 
                              for s in output_parts 
                              for tag in ["GB-ATTACK", "GB-NORMATIVE", "GB-SEX"])
        output_text = " ".join(output_parts)
        
        return input_text, output_text

    def case_partial_bias(self) -> Tuple[str, str]:
        """
        Case 2: PARTIAL BIAS - Mix of biased and neutral
        2-5 sentences, 0-2 with bias tags
        """
        num_sentences = random.randint(2, 5)
        num_biased = random.randint(0, min(2, num_sentences))
        biased_positions = set(random.sample(range(num_sentences), num_biased)) if num_biased > 0 else set()
        
        input_parts = []
        output_parts = []
        
        for i in range(num_sentences):
            if i in biased_positions:
                bias_type = random.choice(list(self.bias_sentences.keys()))
                if self.bias_sentences[bias_type]:
                    sentence = random.choice(self.bias_sentences[bias_type])["text"]
                    input_parts.append(sentence)
                    output_parts.append(self.create_tagged_sentence(sentence, bias_type))
            else:
                if self.neutral_sentences:
                    sentence = random.choice(self.neutral_sentences)
                    input_parts.append(sentence)
                    output_parts.append(sentence)
        
        input_text = " ".join(input_parts)
        output_text = " ".join(output_parts)
        
        return input_text, output_text

    def case_no_bias(self) -> Tuple[str, str]:
        """
        Case 3: NO BIAS - All neutral text
        Multiple sentences, NO tags at all
        """
        num_sentences = random.randint(3, 6)
        output_parts = []
        
        for i in range(num_sentences):
            if self.neutral_sentences:
                sentence = random.choice(self.neutral_sentences)
                output_parts.append(sentence)
        
        output_text = " ".join(output_parts)
        input_text = output_text  # Same as output (no tags)
        
        return input_text, output_text

    def case_long_multi_bias(self) -> Tuple[str, str]:
        """
        Case 4: LONG TEXT WITH MULTIPLE BIASES
        8-15 sentences with 2-4 bias spans scattered throughout
        """
        num_sentences = random.randint(8, 15)
        num_biases = random.randint(2, 4)
        biased_positions = set(random.sample(range(num_sentences), num_biases))
        
        input_parts = []
        output_parts = []
        
        for i in range(num_sentences):
            if i in biased_positions and num_biases > 0:
                bias_type = random.choice(list(self.bias_sentences.keys()))
                if self.bias_sentences[bias_type]:
                    sentence = random.choice(self.bias_sentences[bias_type])["text"]
                    input_parts.append(sentence)
                    output_parts.append(self.create_tagged_sentence(sentence, bias_type))
            else:
                if self.neutral_sentences:
                    sentence = random.choice(self.neutral_sentences)
                    input_parts.append(sentence)
                    output_parts.append(sentence)
        
        input_text = " ".join(input_parts)
        output_text = " ".join(output_parts)
        
        return input_text, output_text

    def case_edge_cases(self) -> Tuple[str, str]:
        """
        Case 5: EDGE CASES & TRICKY NEGATIVES
        Meta-commentary, counter-speech, social critique
        (Should NOT be tagged - they critique bias rather than express it)
        """
        case_type = random.choice(["meta", "counter", "critique"])
        output_parts = []
        
        if case_type == "meta":
            # Meta-commentary: discussing gender bias without expressing it
            if self.meta_sentences:
                sentence = random.choice(self.meta_sentences)
                output_parts.append(sentence)
        elif case_type == "counter":
            # Counter-speech: critiquing gender bias
            critiques = [
                "ยังมีคนพูดว่าผู้หญิงโง่อยู่เลย น่าเสียดายจริง",
                "สังคมไทยชอบโทษผู้หญิงที่โดนคุกคามทางเพศ น่าสมเพช",
                "การบอกว่าผู้ชายต้องเข้มแข็งนี่คือเรื่องโปรแกรม",
                "เห็นคนจำกัดสิทธิผู้หญิงแล้วรู้สึกแย่ จริง ๆ",
            ]
            output_parts.append(random.choice(critiques))
        else:  # critique
            # Social critique about gendered issues
            critiques = [
                "ความเห็นแบบนี้ทำให้สังคมไทยถูกจำกัด",
                "มีความคิดเห็นแบบนี้อยู่จริง แต่มันไม่ถูกต้อง",
                "เรื่องนี้สะท้อนปัญหาสังคมไทยขนาดนี้",
            ]
            output_parts.append(random.choice(critiques))
        
        # Add 1-2 neutral sentences
        for _ in range(random.randint(1, 2)):
            if self.neutral_sentences:
                output_parts.append(random.choice(self.neutral_sentences))
        
        output_text = " ".join(output_parts)
        input_text = output_text  # Same as output (no tags)
        
        return input_text, output_text

    def generate_dataset(self, num_samples: int = 30000) -> List[Dict]:
        """
        Generate balanced dataset with all case types.
        Distribution:
        - Case 1 (Pure Bias): 22% (6,600)
        - Case 2 (Partial Bias): 50% (15,000)
        - Case 3 (No Bias): 22% (6,600)
        - Case 4 (Long Multi): 4% (1,200)
        - Case 5 (Edge Cases): 2% (600)
        """
        print(f"\nGenerating {num_samples} samples with diverse cases...")
        
        # Calculate samples per case
        case1_count = int(num_samples * 0.22)
        case2_count = int(num_samples * 0.50)
        case3_count = int(num_samples * 0.22)
        case4_count = int(num_samples * 0.04)
        case5_count = num_samples - (case1_count + case2_count + case3_count + case4_count)
        
        distribution = {
            CaseType.PURE_BIAS: case1_count,
            CaseType.PARTIAL_BIAS: case2_count,
            CaseType.NO_BIAS: case3_count,
            CaseType.LONG_MULTI: case4_count,
            CaseType.EDGE_CASES: case5_count,
        }
        
        print(f"\nCase Distribution:")
        for case_type, count in distribution.items():
            print(f"  {case_type.value}: {count} samples ({count*100//num_samples}%)")
        
        dataset = []
        case_counts = {case: 0 for case in distribution}
        
        # Generate samples in balanced order
        cases_list = []
        for case_type, count in distribution.items():
            cases_list.extend([case_type] * count)
        random.shuffle(cases_list)
        
        for idx, case_type in enumerate(cases_list):
            try:
                if case_type == CaseType.PURE_BIAS:
                    input_text, output_text = self.case_pure_bias()
                elif case_type == CaseType.PARTIAL_BIAS:
                    input_text, output_text = self.case_partial_bias()
                elif case_type == CaseType.NO_BIAS:
                    input_text, output_text = self.case_no_bias()
                elif case_type == CaseType.LONG_MULTI:
                    input_text, output_text = self.case_long_multi_bias()
                else:  # EDGE_CASES
                    input_text, output_text = self.case_edge_cases()
                
                if input_text and output_text:
                    sample = {
                        "instruction": INSTRUCTION,
                        "input": input_text,
                        "output": output_text,
                        "case_type": case_type.value,  # For analysis
                    }
                    dataset.append(sample)
                    case_counts[case_type] += 1
                
                if (idx + 1) % 5000 == 0:
                    print(f"  Generated {idx + 1}/{num_samples} samples")
            
            except Exception as e:
                print(f"Warning: Skipped sample at {idx}: {e}")
                continue
        
        print(f"\nGenerated {len(dataset)} valid samples")
        print("\nActual Distribution:")
        for case_type, count in case_counts.items():
            print(f"  {case_type.value}: {count} samples")
        
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
        
        return str(output_file)

    def split_dataset(self, dataset: List[Dict], train_ratio: float = 0.95) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and validation sets (stratified by case type)."""
        # Stratified split by case type
        train_data = []
        val_data = []
        
        case_types = set(s["case_type"] for s in dataset)
        
        for case_type in case_types:
            case_samples = [s for s in dataset if s["case_type"] == case_type]
            split_idx = int(len(case_samples) * train_ratio)
            train_data.extend(case_samples[:split_idx])
            val_data.extend(case_samples[split_idx:])
        
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        return train_data, val_data

    def save_split_datasets(self, dataset: List[Dict], output_dir: str, train_ratio: float = 0.95) -> Tuple[str, str]:
        """Save train and validation datasets separately."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_data, val_data = self.split_dataset(dataset, train_ratio)
        
        train_file = output_dir / "instruction_train.jsonl"
        val_file = output_dir / "instruction_val.jsonl"
        
        with open(train_file, "w", encoding="utf-8") as f:
            for sample in train_data:
                # Remove case_type before saving (only for generation analysis)
                sample_copy = {k: v for k, v in sample.items() if k != "case_type"}
                f.write(json.dumps(sample_copy, ensure_ascii=False) + "\n")
        
        with open(val_file, "w", encoding="utf-8") as f:
            for sample in val_data:
                sample_copy = {k: v for k, v in sample.items() if k != "case_type"}
                f.write(json.dumps(sample_copy, ensure_ascii=False) + "\n")
        
        print(f"\nDatasets saved:")
        print(f"  Train: {train_file} ({len(train_data)} samples)")
        print(f"  Val: {val_file} ({len(val_data)} samples)")
        
        return str(train_file), str(val_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse instruction-input-output format training data for gender bias detection"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30000,
        help="Number of samples to generate (default: 30000)",
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
    generator = EnhancedInstructionDataGenerator(seed=args.seed)

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
