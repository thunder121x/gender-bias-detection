#!/usr/bin/env python3
"""
Simple fine-tuning script for Gender Bias Detection
Run this after stopping vLLM to free up memory

Usage:
    python3 train_simple.py --preset balanced
    python3 train_simple.py --preset memory_constrained
    python3 train_simple.py --preset high_memory
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from h100_config import H100OptimizedConfig, H100Presets

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GenderBiasDataset:
    """Load gender bias dataset from JSONL files."""
    
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        
        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example.get("messages", [])
        
        # Format conversation
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


def setup_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {params / 1e9:.2f}B parameters")
    
    return model, tokenizer


def setup_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """Apply LoRA to model."""
    logger.info("Applying LoRA...")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train(
    config: H100OptimizedConfig,
    train_file: str,
    val_file: str,
):
    """Main training function."""
    
    set_seed(config.seed)
    
    logger.info("=" * 80)
    logger.info("GENDER BIAS DETECTION - FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"GPU: H100 NVL (95GB VRAM)")
    logger.info(f"Memory optimization: Gradient checkpointing + Gradient accumulation")
    logger.info(f"Batch size: {config.per_device_train_batch_size} (per device)")
    logger.info(f"Accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info("=" * 80)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config.model_name)
    model = setup_lora(model, config.lora_r, config.lora_alpha, config.lora_dropout)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = GenderBiasDataset(train_file, tokenizer, config.max_seq_length)
    eval_dataset = GenderBiasDataset(val_file, tokenizer, config.max_seq_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        max_grad_norm=config.max_grad_norm,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        bf16=config.bf16,
        fp16=config.fp16,
        seed=config.seed,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
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
    trainer.save_model(training_args.output_dir)
    
    # Summary
    summary = {
        "model": config.model_name,
        "training_time_hours": train_result.training_time / 3600,
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics,
        "config": {
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "per_device_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_path = Path(training_args.output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune for gender bias detection on H100")
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["memory_constrained", "balanced", "high_memory"],
        help="Training preset (memory vs speed tradeoff)"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="services/lora_finetuning/training_data/train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="services/lora_finetuning/training_data/val.jsonl",
        help="Path to validation data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat",
        help="Model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lora_checkpoints/gender_bias_h100",
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Get preset config
    preset_map = {
        "memory_constrained": H100Presets.memory_constrained,
        "balanced": H100Presets.balanced,
        "high_memory": H100Presets.high_memory,
    }
    
    config = preset_map[args.preset]()
    config.model_name = args.model
    config.output_dir = args.output_dir
    
    # Verify data files exist
    if not Path(args.train_file).exists():
        logger.error(f"Train file not found: {args.train_file}")
        return
    if not Path(args.val_file).exists():
        logger.error(f"Val file not found: {args.val_file}")
        return
    
    # Save config
    config_path = Path(args.output_dir) / "training_config.json"
    config.to_json(str(config_path))
    logger.info(f"Config saved to: {config_path}")
    
    # Train
    train(config, args.train_file, args.val_file)


if __name__ == "__main__":
    main()
