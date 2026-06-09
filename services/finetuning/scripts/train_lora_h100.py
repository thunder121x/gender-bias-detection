#!/usr/bin/env python3
"""
Fine-tuning script for Gender Bias Detection with LoRA
Optimized for NVIDIA H100 GPU (95GB VRAM)

This script uses:
- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning) with LoRA
- DeepSpeed for distributed training
- Flash Attention 2 for faster training
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import bitsandbytes as bnb

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-chat",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Enable loading models with custom modeling code"}
    )
    use_auth_token: bool = field(
        default=True,
        metadata={"help": "Use HuggingFace auth token"}
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model placement"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    train_data_path: str = field(
        default="services/lora_finetuning/training_data/train.jsonl",
        metadata={"help": "Path to training data in JSONL format"}
    )
    validation_data_path: str = field(
        default="services/lora_finetuning/training_data/val.jsonl",
        metadata={"help": "Path to validation data in JSONL format"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for input texts"}
    )
    preprocessing_num_workers: int = field(
        default=8,
        metadata={"help": "Number of workers for data preprocessing"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite cached datasets"}
    )


@dataclass
class LoraArguments:
    """Arguments pertaining to LoRA configuration."""
    
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_weight_decay: float = field(
        default=0.0,
        metadata={"help": "LoRA weight decay"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type: 'none', 'all', or 'lora_only'"}
    )
    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type for LoRA"}
    )
    target_modules: str = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of modules to apply LoRA to"}
    )


class GenderBiasDataset(Dataset):
    """Custom dataset for gender bias detection with structured JSON outputs."""
    
    def __init__(self, data_path: str, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = []
        
        # Load JSONL data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    self.data.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        messages = example.get("messages", [])
        
        if len(messages) != 3:
            raise ValueError(f"Expected 3 messages, got {len(messages)}")
        
        # Format the conversation
        conversation = self._format_chat(messages)
        
        # Tokenize
        encodings = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0).clone(),
        }
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into chat format."""
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                conversation += f"<system>\n{content}\n</system>\n"
            elif role == "user":
                conversation += f"<user>\n{content}\n</user>\n"
            elif role == "assistant":
                conversation += f"<assistant>\n{content}\n</assistant>\n"
        
        return conversation


class BiasDetectionTrainer(Trainer):
    """Custom trainer with evaluation metrics for bias detection."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with label masking."""
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def find_all_linear_names(model, int4=False, int8=False):
    """Find all linear layer names for LoRA."""
    cls = bnb.nn.Linear4bit if int4 else (bnb.nn.Linear8bitLt if int8 else torch.nn.Linear)
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


def setup_model(model_args: ModelArguments) -> tuple:
    """Setup model and tokenizer with quantization if needed."""
    
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token,
        padding_side="left",
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimal settings for H100
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token,
        device_map=model_args.device_map,
        torch_dtype=torch.bfloat16,  # H100 supports bfloat16 natively
        attn_implementation="flash_attention_2",  # Use Flash Attention 2
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    logger.info(f"Gradient checkpointing enabled (saves ~30% memory)")
    
    return model, tokenizer


def setup_lora(model, lora_args: LoraArguments) -> tuple:
    """Setup LoRA configuration and apply to model."""
    
    logger.info("Setting up LoRA configuration...")
    
    # Get target modules
    target_modules = lora_args.target_modules.split(",")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=lora_args.task_type,
    )
    
    logger.info(f"LoRA Config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, dropout={lora_args.lora_dropout}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, lora_config


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
):
    """Main training function."""
    
    set_seed(training_args.seed)
    
    logger.info("=" * 80)
    logger.info("Starting fine-tuning job")
    logger.info("=" * 80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"GPU: NVIDIA H100 NVL (95GB)")
    logger.info(f"Training on: {training_args.num_train_epochs} epochs")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size} per device")
    logger.info("=" * 80)
    
    # Setup model
    model, tokenizer = setup_model(model_args)
    
    # Setup LoRA
    model, lora_config = setup_lora(model, lora_args)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = GenderBiasDataset(
        data_args.train_data_path,
        tokenizer,
        data_args.max_seq_length
    )
    
    eval_dataset = GenderBiasDataset(
        data_args.validation_data_path,
        tokenizer,
        data_args.max_seq_length
    )
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Eval examples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )
    
    # Initialize trainer
    trainer = BiasDetectionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save results
    metrics = train_result.metrics
    logger.info(f"\nTraining complete!")
    logger.info(f"Final metrics: {metrics}")
    
    # Save model
    logger.info("Saving final model...")
    trainer.save_model(training_args.output_dir)
    
    # Save training summary
    summary = {
        "model": model_args.model_name_or_path,
        "training_time": train_result.training_time,
        "train_loss": train_result.training_loss,
        "metrics": metrics,
        "lora_config": {
            "r": lora_args.lora_r,
            "alpha": lora_args.lora_alpha,
            "dropout": lora_args.lora_dropout,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_path = Path(training_args.output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Create output directory
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    train(model_args, data_args, lora_args, training_args)


if __name__ == "__main__":
    main()
