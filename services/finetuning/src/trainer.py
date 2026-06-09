"""
Fine-tuning script for token classification model using HuggingFace Transformers.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict, load_dataset

import sys
sys.path.append(os.path.dirname(__file__))

from dataset_processor import TokenClassificationDataset


logger = logging.getLogger(__name__)


class TokenClassificationTrainer:
    """Handles model fine-tuning for token classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.dataset_processor = None
        
    def setup(self) -> None:
        """Initialize tokenizer and model."""
        print(f"Loading tokenizer and model: {self.config['model']['base_model']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['base_model'],
            trust_remote_code=True
        )
        
        # Load model
        num_labels = len(self.config['token_classification']['labels'])
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config['model']['base_model'],
            num_labels=num_labels,
            id2label=self.config['token_classification']['id2label'],
            label2id=self.config['token_classification']['label2id'],
            trust_remote_code=True
        )
        
        # Setup dataset processor
        self.dataset_processor = TokenClassificationDataset(
            self.config, 
            self.tokenizer
        )
        
        print(f"Model loaded successfully!")
        print(f"Number of parameters: {self.model.num_parameters():,}")
    
    def load_and_prepare_datasets(self, train_path: str, val_path: str, 
                                  test_path: str) -> None:
        """Load and prepare datasets."""
        print("\nPreparing datasets...")
        
        # Load datasets
        train_dataset = load_dataset('json', data_files=train_path)['train']
        val_dataset = load_dataset('json', data_files=val_path)['train']
        test_dataset = load_dataset('json', data_files=test_path)['train']
        
        print(f"Loaded datasets:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        # Tokenize and align labels
        def tokenize_and_align(examples):
            tokenized_inputs = self.tokenizer(
                examples['text'],
                truncation=True,
                is_split_into_words=False,
                max_length=self.config['model']['max_length'],
                padding='max_length',
            )
            
            labels = []
            for i, label in enumerate(examples['token_labels']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        if word_idx < len(label):
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(0)
                    else:
                        if word_idx < len(label):
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(0)
                    
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs['labels'] = labels
            return tokenized_inputs
        
        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        val_dataset = val_dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset"
        )
        
        test_dataset = test_dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test dataset"
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        print("Datasets prepared successfully!")
    
    def compute_metrics(self, p):
        """Compute evaluation metrics."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.config['token_classification']['id2label'][p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.config['token_classification']['id2label'][l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Simple metrics calculation
        tp = sum(1 for pred_seq, true_seq in zip(true_predictions, true_labels) 
                for p, t in zip(pred_seq, true_seq) if p == 'B-BIAS' and t == 'B-BIAS')
        fp = sum(1 for pred_seq, true_seq in zip(true_predictions, true_labels) 
                for p, t in zip(pred_seq, true_seq) if p == 'B-BIAS' and t != 'B-BIAS')
        fn = sum(1 for pred_seq, true_seq in zip(true_predictions, true_labels) 
                for p, t in zip(pred_seq, true_seq) if p != 'B-BIAS' and t == 'B-BIAS')
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self) -> None:
        """Train the model."""
        print("\nStarting training...")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['save_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            logging_dir=self.config['training']['log_dir'],
            logging_steps=self.config['training']['logging_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_total_limit=self.config['training']['save_total_limit'],
            evaluation_strategy="epoch",
            learning_rate=self.config['training']['learning_rate'],
            push_to_hub=False,
            seed=42
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # Train
        self.trainer.train()
        
        print("Training completed!")
    
    def evaluate(self) -> Dict:
        """Evaluate model on test set."""
        print("\nEvaluating on test set...")
        
        if self.trainer is None:
            raise ValueError("Must call train() first")
        
        results = self.trainer.evaluate(self.test_dataset)
        
        print("Test Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def save_model(self, save_dir: str = None) -> None:
        """Save model and tokenizer."""
        if save_dir is None:
            save_dir = self.config['training']['save_dir']
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config
        with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Model saved to {save_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load saved model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        
        print(f"Model loaded from {model_dir}")
