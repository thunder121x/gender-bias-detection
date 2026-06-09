"""
Dataset processor: Convert augmented paragraphs into token classification format
compatible with HuggingFace Transformers.
"""

import json
import os
from typing import List, Dict, Tuple
from pathlib import Path


class TokenClassificationDataset:
    """Processes data for token-level classification with BIO tags."""
    
    def __init__(self, config: Dict, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.label2id = config['token_classification']['label2id']
        self.id2label = config['token_classification']['id2label']
        
    def load_from_jsonl(self, filepath: str) -> List[Dict]:
        """Load dataset from JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def align_labels_with_tokens(self, tokenized_inputs, word_ids: List[int], 
                                  sentence_labels: List[int]) -> List[int]:
        """
        Align sentence-level labels with subword tokens.
        
        When text is tokenized with subword tokenization (e.g., WordPiece),
        we need to map token positions back to words/sentences.
        
        Args:
            tokenized_inputs: Output from tokenizer
            word_ids: Mapping of tokens to word positions
            sentence_labels: Labels for each word
        
        Returns:
            Token-level labels aligned with subword tokens
        """
        labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                labels.append(-100)  # Ignore special tokens in loss calculation
            elif word_idx != previous_word_idx:
                # First token of a word
                labels.append(self.label2id[self._get_label_for_word(
                    word_idx, sentence_labels
                )])
            else:
                # Continuation of previous word's token
                # Keep the same label (B-BIAS stays B-BIAS, I-BIAS becomes I-BIAS)
                labels.append(self.label2id[self._get_label_for_word(
                    word_idx, sentence_labels
                )])
            
            previous_word_idx = word_idx
        
        return labels
    
    def _get_label_for_word(self, word_idx: int, sentence_labels: List[int]) -> str:
        """Get BIO label for a word based on sentence labels."""
        if word_idx >= len(sentence_labels):
            return "O"
        
        if sentence_labels[word_idx] == 1:
            return "B-BIAS"  # Simplified: all biased words get B-BIAS
        return "O"
    
    def tokenize_and_align_labels(self, examples: List[Dict], 
                                   max_length: int = 512) -> Dict:
        """
        Tokenize text and align labels with tokens.
        
        Args:
            examples: List of examples with 'text' and 'token_labels'
            max_length: Maximum sequence length
        
        Returns:
            Tokenized batch with aligned labels
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        tokenized_inputs = self.tokenizer(
            [ex['text'] for ex in examples],
            truncation=True,
            is_split_into_words=False,
            max_length=max_length,
            padding='max_length',
            return_offsets_mapping=True
        )
        
        labels = []
        
        for i, example in enumerate(examples):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            token_labels = example['token_labels']
            
            # Align labels with tokens
            aligned_labels = self.align_labels_with_tokens(
                tokenized_inputs, word_ids, token_labels
            )
            labels.append(aligned_labels)
        
        tokenized_inputs['labels'] = labels
        
        # Remove offset mapping as we don't need it in final dataset
        tokenized_inputs.pop('offset_mapping', None)
        
        return tokenized_inputs
    
    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer instance."""
        self.tokenizer = tokenizer
    
    def prepare_dataset(self, data: List[Dict], max_length: int = 512) -> List[Dict]:
        """
        Prepare dataset for training.
        
        Args:
            data: List of raw examples
            max_length: Maximum sequence length
        
        Returns:
            List of processed examples with tokenized inputs and labels
        """
        processed_data = []
        
        for example in data:
            if self.tokenizer is None:
                # If no tokenizer, use simple space-based tokenization
                tokens = example['text'].split()
                token_labels = example.get('token_labels', 
                                          [0] * len(tokens))
            else:
                # Use proper tokenization
                encoding = self.tokenizer(
                    example['text'],
                    truncation=True,
                    max_length=max_length,
                    return_offsets_mapping=True
                )
                
                word_ids = encoding.word_ids()
                token_labels = example.get('token_labels', [0])
                
                aligned_labels = self.align_labels_with_tokens(
                    encoding, word_ids, token_labels
                )
                
                processed_example = {
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'token_type_ids': encoding.get('token_type_ids', [0] * len(encoding['input_ids'])),
                    'labels': aligned_labels,
                    'text': example['text'],
                    'sentence_labels': example.get('sentence_labels', []),
                    'bias_info': example.get('bias_info', [])
                }
                
                processed_data.append(processed_example)
                continue
            
            # Simple tokenization fallback
            processed_example = {
                'tokens': tokens,
                'labels': token_labels,
                'text': example['text'],
                'sentence_labels': example.get('sentence_labels', []),
                'bias_info': example.get('bias_info', [])
            }
            
            processed_data.append(processed_example)
        
        return processed_data
    
    def save_processed_dataset(self, data: List[Dict], output_path: str) -> None:
        """Save processed dataset to JSONL file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(data)} processed examples to {output_path}")
    
    def get_label_stats(self, data: List[Dict]) -> Dict:
        """Calculate label statistics."""
        label_counts = {label: 0 for label in self.id2label.values()}
        total_tokens = 0
        
        for example in data:
            labels = example.get('labels', [])
            for label_id in labels:
                if label_id >= 0:  # Ignore -100 (special tokens)
                    label_counts[self.id2label.get(label_id, 'UNKNOWN')] += 1
                    total_tokens += 1
        
        return {
            'total_tokens': total_tokens,
            'label_counts': label_counts,
            'label_distribution': {
                label: count / total_tokens if total_tokens > 0 else 0
                for label, count in label_counts.items()
            }
        }
