"""
Data augmentation pipeline: Generate synthetic paragraphs with sentence-level bias labels
by combining GB and NON-GB sentences at random positions.
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class DataAugmenter:
    """Augments data by combining sentences into paragraphs with BIO labels."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gb_sentences = []
        self.non_gb_sentences = []
        
    def load_sentences(self, input_dir: str) -> None:
        """Load all GB and NON-GB sentences from label files."""
        print(f"Loading sentences from {input_dir}...")
        
        # Load GB sentences
        gb_types = ["gb_attack.json", "gb_normative.json", "gb_sex.json"]
        for gb_file in gb_types:
            filepath = os.path.join(input_dir, gb_file)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
                    self.gb_sentences.extend(sentences)
                    print(f"  Loaded {len(sentences)} sentences from {gb_file}")
        
        # Load NON-GB sentences
        non_gb_types = ["non_gb_insult.json", "non_gb_meta.json", "non_gb_neutral.json"]
        for non_gb_file in non_gb_types:
            filepath = os.path.join(input_dir, non_gb_file)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
                    self.non_gb_sentences.extend(sentences)
                    print(f"  Loaded {len(sentences)} sentences from {non_gb_file}")
        
        print(f"Total GB sentences: {len(self.gb_sentences)}")
        print(f"Total NON-GB sentences: {len(self.non_gb_sentences)}")
    
    def _sample_sentences(self, sentences: List[Dict], count: int, 
                         exclude_indices: set = None) -> List[Dict]:
        """Randomly sample sentences without replacement."""
        exclude_indices = exclude_indices or set()
        available = [s for i, s in enumerate(sentences) if i not in exclude_indices]
        return random.sample(available, min(count, len(available)))
    
    def _generate_paragraph(self, num_gb: int = 1) -> Dict:
        """
        Generate a single paragraph with specified number of GB sentences.
        
        Returns:
            {
                'text': 'sentence1. sentence2. sentence3...',
                'sentences': ['sentence1', 'sentence2', 'sentence3'],
                'token_labels': [0, 0, 1, 1, 0, 0, ...],  # BIO tags for each token
                'sentence_labels': [0, 0, 1],  # 0=non-bias, 1=biased
                'bias_info': [
                    {'text': '...', 'index': 0, 'subtype': 'GB-ATTACK', 'target': '...'}
                ]
            }
        """
        # Determine paragraph structure
        min_sents = self.config['data']['paragraph']['min_sentences']
        max_sents = self.config['data']['paragraph']['max_sentences']
        total_sentences = random.randint(min_sents, max_sents)
        
        num_non_gb = total_sentences - num_gb
        num_non_gb = max(1, num_non_gb)  # At least 1 non-GB sentence
        
        # Sample sentences
        gb_samples = self._sample_sentences(self.gb_sentences, num_gb)
        non_gb_samples = self._sample_sentences(self.non_gb_sentences, num_non_gb)
        
        # Combine and shuffle
        all_sentences = gb_samples + non_gb_samples
        random.shuffle(all_sentences)
        
        # Build paragraph text and labels
        sentence_texts = [s['text'] for s in all_sentences]
        paragraph_text = " ".join(sentence_texts)
        
        # Create sentence-level labels (0=non-bias, 1=bias)
        sentence_labels = []
        bias_info = []
        
        for idx, sent_obj in enumerate(all_sentences):
            is_bias = sent_obj['label'] == 'GB'
            sentence_labels.append(1 if is_bias else 0)
            
            if is_bias:
                bias_info.append({
                    'text': sent_obj['text'],
                    'index': idx,
                    'subtype': sent_obj.get('subtype', 'GB-UNKNOWN'),
                    'target': sent_obj.get('bias_target', 'unknown')
                })
        
        # Convert to token-level BIO labels
        token_labels = self._create_token_labels(
            sentence_texts, 
            sentence_labels
        )
        
        return {
            'text': paragraph_text,
            'sentences': sentence_texts,
            'sentence_labels': sentence_labels,
            'token_labels': token_labels,
            'bias_info': bias_info,
            'num_sentences': len(sentence_texts)
        }
    
    def _create_token_labels(self, sentence_texts: List[str], 
                            sentence_labels: List[int]) -> List[int]:
        """
        Convert sentence-level labels to token-level BIO labels.
        
        Args:
            sentence_texts: List of sentence texts
            sentence_labels: List of sentence labels (0 or 1)
        
        Returns:
            List of token labels (0=O, 1=B-BIAS, 2=I-BIAS)
        """
        token_labels = []
        first_token_in_sentence = True
        
        for sent_text, sent_label in zip(sentence_texts, sentence_labels):
            # Simple tokenization by spaces (assuming Thai text with spaces)
            tokens = sent_text.split()
            
            for token_idx, token in enumerate(tokens):
                if sent_label == 1:  # Biased sentence
                    if first_token_in_sentence:
                        token_labels.append(1)  # B-BIAS
                        first_token_in_sentence = False
                    else:
                        token_labels.append(2)  # I-BIAS
                else:  # Non-biased sentence
                    token_labels.append(0)  # O
            
            first_token_in_sentence = True
        
        return token_labels
    
    def generate_dataset(self, num_samples: int = 10000) -> List[Dict]:
        """
        Generate synthetic dataset with specified bias distribution.
        
        Args:
            num_samples: Total number of paragraphs to generate
        
        Returns:
            List of generated paragraphs with labels
        """
        if not self.gb_sentences or not self.non_gb_sentences:
            raise ValueError("Must call load_sentences() first")
        
        print(f"\nGenerating {num_samples} synthetic paragraphs...")
        
        distribution = self.config['data']['paragraph']['distribution']
        
        # Calculate how many paragraphs for each category
        no_bias_count = int(num_samples * distribution['no_bias_ratio'])
        one_bias_count = int(num_samples * distribution['one_bias_ratio'])
        two_bias_count = int(num_samples * distribution['two_bias_ratio'])
        three_bias_count = num_samples - no_bias_count - one_bias_count - two_bias_count
        
        dataset = []
        
        # Generate paragraphs with no bias
        print(f"  Generating {no_bias_count} paragraphs with no bias...")
        for _ in range(no_bias_count):
            para = self._generate_paragraph(num_gb=0)
            dataset.append(para)
        
        # Generate paragraphs with 1 GB sentence
        print(f"  Generating {one_bias_count} paragraphs with 1 bias...")
        for _ in range(one_bias_count):
            para = self._generate_paragraph(num_gb=1)
            dataset.append(para)
        
        # Generate paragraphs with 2 GB sentences
        print(f"  Generating {two_bias_count} paragraphs with 2 biases...")
        for _ in range(two_bias_count):
            para = self._generate_paragraph(num_gb=2)
            dataset.append(para)
        
        # Generate paragraphs with 3+ GB sentences
        print(f"  Generating {three_bias_count} paragraphs with 3+ biases...")
        for _ in range(three_bias_count):
            num_gb = random.randint(3, 5)
            para = self._generate_paragraph(num_gb=min(num_gb, len(self.gb_sentences)))
            dataset.append(para)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        print(f"Generated {len(dataset)} paragraphs total")
        
        # Print statistics
        self._print_statistics(dataset)
        
        return dataset
    
    def _print_statistics(self, dataset: List[Dict]) -> None:
        """Print dataset statistics."""
        total_paragraphs = len(dataset)
        paragraphs_with_bias = sum(1 for p in dataset if any(p['sentence_labels']))
        total_biased_sentences = sum(sum(p['sentence_labels']) for p in dataset)
        avg_biased_per_biased_para = (
            total_biased_sentences / paragraphs_with_bias 
            if paragraphs_with_bias > 0 else 0
        )
        
        print(f"\n  Dataset Statistics:")
        print(f"    Total paragraphs: {total_paragraphs}")
        print(f"    Paragraphs with bias: {paragraphs_with_bias} ({100*paragraphs_with_bias/total_paragraphs:.1f}%)")
        print(f"    Total biased sentences: {total_biased_sentences}")
        print(f"    Avg biased sentences per paragraph (with bias): {avg_biased_per_biased_para:.2f}")
    
    def save_dataset(self, dataset: List[Dict], output_dir: str) -> Tuple[str, str, str]:
        """
        Save dataset split into train, validation, and test sets.
        
        Returns:
            Tuple of (train_path, val_path, test_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Split dataset
        train_split = self.config['data']['train_test_split']
        val_split = self.config['data']['validation_split']
        
        train_size = int(len(dataset) * train_split * (1 - val_split))
        val_size = int(len(dataset) * train_split * val_split)
        
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]
        
        # Save as JSONL for compatibility with HuggingFace
        train_path = os.path.join(output_dir, 'train.jsonl')
        val_path = os.path.join(output_dir, 'validation.jsonl')
        test_path = os.path.join(output_dir, 'test.jsonl')
        
        for path, data in [(train_path, train_data), (val_path, val_data), (test_path, test_data)]:
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved {len(data)} samples to {path}")
        
        return train_path, val_path, test_path
