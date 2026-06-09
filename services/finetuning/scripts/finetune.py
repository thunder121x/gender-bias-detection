#!/usr/bin/env python3
"""
Simple one-file fine-tuning script for Gender Bias Detection
Optimized for H100 GPU

Usage:
    python3 finetune.py --train-file path/to/train.jsonl \
                        --val-file path/to/val.jsonl
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GenderBiasDataset(Dataset):
    """Load gender bias dataset from JSONL files with proper chat template formatting."""
    
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading dataset from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    self.examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example.get("messages", [])
        
        # Use tokenizer.apply_chat_template for proper format
        # This ensures the format matches the model's training format
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            # Fallback if apply_chat_template not available
            logger.warning(f"apply_chat_template failed: {e}, using manual format")
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    text += f"<|system|>\n{content}\n"
                elif role == "user":
                    text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}\n"
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze().clone(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gender Bias Detection model on H100"
    )
    
    # Data arguments
    parser.add_argument(
        "--train-file",
        type=str,
        default="services/lora_finetuning/training_data/train_inline_tags_v2.jsonl",
        help="Path to training data (inline tags format)"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="services/lora_finetuning/training_data/val_inline_tags_v2.jsonl",
        help="Path to validation data (inline tags format)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat",
        help="Base model name or path"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per device"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lora_checkpoints/gender_bias_h100",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Verify data files exist
    for file_path in [args.train_file, args.val_file]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return
    
    logger.info("=" * 80)
    logger.info("GENDER BIAS DETECTION - FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU: H100 NVL (95GB VRAM)")
    logger.info(f"Training data: {args.train_file}")
    logger.info(f"Validation data: {args.val_file}")
    logger.info(f"Batch size: {args.batch_size} (per device)")
    logger.info(f"Accumulation steps: {args.accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Chat template: Using tokenizer.apply_chat_template")
    logger.info(f"Labels format: Inline tags (not JSON)")
    logger.info("=" * 80)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled (saves ~30% memory)")
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = GenderBiasDataset(
        args.train_file,
        tokenizer,
        args.max_seq_length
    )
    
    eval_dataset = GenderBiasDataset(
        args.val_file,
        tokenizer,
        args.max_seq_length
    )
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Eval examples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        bf16=True,
        fp16=False,
        seed=args.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80 + "\n")
    
    train_result = trainer.train()
    
    # Save
    logger.info("\nSaving model...")
    trainer.save_model(args.output_dir)
    
    # Summary
    summary = {
        "model": args.model,
        "training_time_hours": train_result.training_time / 3600,
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics,
        "config": {
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "lora_r": 64,
            "lora_alpha": 128,
            "chat_template": "tokenizer.apply_chat_template",
            "labels_format": "inline_tags",
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
