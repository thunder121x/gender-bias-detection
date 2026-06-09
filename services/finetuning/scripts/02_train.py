#!/usr/bin/env python3
"""
Gender Bias Detection - Token Classification Fine-tuning
Converts the Jupyter notebook to a runnable script for efficient training.
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict, load_dataset


def main():
    # ========================================
    # Step 1: Setup & Configuration
    # ========================================
    print("="*80)
    print("STEP 1: SETUP & CONFIGURATION")
    print("="*80)
    
    # Determine paths
    script_dir = Path(__file__).parent
    finetuning_dir = script_dir.parent
    
    DRIVE_TRAIN_FILE = finetuning_dir / "data" / "train.jsonl"
    DRIVE_VAL_FILE = finetuning_dir / "data" / "validation.jsonl"
    DRIVE_TEST_FILE = finetuning_dir / "data" / "test.jsonl"
    MODEL_SAVE_DIR = finetuning_dir / "models"
    LOG_DIR = finetuning_dir / "logs"
    
    # Create directories if they don't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"Training data: {DRIVE_TRAIN_FILE}")
    print(f"Validation data: {DRIVE_VAL_FILE}")
    print(f"Test data: {DRIVE_TEST_FILE}")
    print(f"Model save dir: {MODEL_SAVE_DIR}")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ========================================
    # Step 2: Check & Load Data
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: LOAD DATA")
    print("="*80)
    
    for file_path in [DRIVE_TRAIN_FILE, DRIVE_VAL_FILE, DRIVE_TEST_FILE]:
        if file_path.exists():
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"✓ {file_path.name}: {line_count} samples")
        else:
            print(f"✗ {file_path} NOT FOUND")
            sys.exit(1)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset('json', data_files=str(DRIVE_TRAIN_FILE))['train']
    val_dataset = load_dataset('json', data_files=str(DRIVE_VAL_FILE))['train']
    test_dataset = load_dataset('json', data_files=str(DRIVE_TEST_FILE))['train']
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Show sample
    print(f"\nSample:")
    sample = train_dataset[0]
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Sentences: {len(sample['sentences'])}")
    print(f"  Has bias: {any(sample['sentence_labels'])}")
    
    # ========================================
    # Step 3: Load Model & Tokenizer
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: LOAD MODEL & TOKENIZER")
    print("="*80)
    
    model_name = "xlm-roberta-base"
    num_labels = 3  # O, B-BIAS, I-BIAS
    id2label = {0: "O", 1: "B-BIAS", 2: "I-BIAS"}
    label2id = {"O": 0, "B-BIAS": 1, "I-BIAS": 2}
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True
    )
    
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Device: {next(model.parameters()).device}")
    
    # ========================================
    # Step 4: Tokenize & Align Labels
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: TOKENIZE & ALIGN LABELS")
    print("="*80)
    
    def tokenize_and_align_labels(examples, max_length=512):
        """Tokenize text and align labels with tokens."""
        tokenized_inputs = tokenizer(
            examples['text'],
            truncation=True,
            is_split_into_words=False,
            max_length=max_length,
            padding='max_length',
        )
        
        labels = []
        for i, label in enumerate(examples['token_labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:  # Special tokens
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
    
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    val_tokenized = val_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset"
    )
    
    test_tokenized = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test dataset"
    )
    
    print(f"✓ Tokenization complete")
    
    # ========================================
    # Step 5: Setup Training
    # ========================================
    print("\n" + "="*80)
    print("STEP 5: SETUP TRAINING")
    print("="*80)
    
    def compute_metrics(p):
        """Compute evaluation metrics."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
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
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(LOG_DIR),
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        push_to_hub=False,
        seed=42,
        disable_tqdm=False,  # Show progress bar
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    print("✓ Training setup complete")
    
    # ========================================
    # Step 6: Train Model
    # ========================================
    print("\n" + "="*80)
    print("STEP 6: TRAIN MODEL")
    print("="*80)
    print(f"Total training samples: {len(train_tokenized)}")
    print(f"Total validation samples: {len(val_tokenized)}")
    print(f"Batch size: 16")
    print(f"Epochs: 3")
    print()
    
    train_result = trainer.train()
    
    print("\n✓ Training complete!")
    print(f"Training time: {train_result.training_time / 3600:.2f} hours")
    
    # ========================================
    # Step 7: Evaluate on Test Set
    # ========================================
    print("\n" + "="*80)
    print("STEP 7: EVALUATE ON TEST SET")
    print("="*80)
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_tokenized)
    
    print("\nTest Results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # ========================================
    # Step 8: Save Model
    # ========================================
    print("\n" + "="*80)
    print("STEP 8: SAVE MODEL")
    print("="*80)
    
    model_save_path = MODEL_SAVE_DIR / 'final_model'
    model.save_pretrained(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    
    # Save training config
    config = {
        'model_name': model_name,
        'num_labels': num_labels,
        'id2label': id2label,
        'label2id': label2id,
        'training_time_hours': train_result.training_time / 3600,
        'test_results': test_results
    }
    
    with open(model_save_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Model saved to {model_save_path}")
    print(f"\nFiles:")
    for file in sorted(os.listdir(model_save_path)):
        file_path = model_save_path / file
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / 1e6  # MB
            print(f"  - {file} ({size:.1f} MB)")
    
    # ========================================
    # Step 9: Summary
    # ========================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    print(f"\nModel: {model_name}")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"\nTraining Summary:")
    print(f"  Samples: {len(train_tokenized)} train, {len(val_tokenized)} val, {len(test_tokenized)} test")
    print(f"  Epochs: 3")
    print(f"  Time: {train_result.training_time / 3600:.2f} hours")
    
    print(f"\nTest Results:")
    print(f"  Precision: {test_results.get('eval_precision', 0):.4f}")
    print(f"  Recall: {test_results.get('eval_recall', 0):.4f}")
    print(f"  F1: {test_results.get('eval_f1', 0):.4f}")
    
    print(f"\nModel saved to: {model_save_path}")
    print("\n✓ Next step: Use the model for inference!")


if __name__ == "__main__":
    main()
