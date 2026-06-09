#!/usr/bin/env python3
"""
Qwen 3.5 2B LoRA Fine-tuning on RTX 6000 Blackwell
Optimized for Thai Gender Bias Span Detection

Uses Unsloth for 2-5x faster training with 16-bit LoRA (not QLoRA).
RTX 6000 has 96GB VRAM, so we use full 16-bit precision for span detection accuracy.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"  # Disable weights & biases if not needed

print("=" * 80)
print("INITIALIZING UNSLOTH + QWEN 3.5 2B LoRA FINE-TUNING")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Using 3B as 2B may not exist
    max_seq_length: int = 4096
    
    # LoRA config (optimized for RTX 6000)
    lora_rank: int = 128  # High rank for linguistic nuance
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    
    # Training
    batch_size: int = 64  # Can afford high batch size on 96GB VRAM
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_steps: int = 100
    
    # Paths — training_data/ is restored from the project Google Drive
    # (see root README, Data & Models) before running
    train_file: str = "training_data/train.jsonl"
    val_file: str = "training_data/val.jsonl"
    output_dir: str = "qwen_gb_detector_lora"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_flash_attention: bool = True
    
    def __post_init__(self):
        if self.device == "cuda":
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"⚠️  CUDA not available, using CPU (will be slow)")


# ============================================================================
# INSTALL & IMPORT
# ============================================================================

def install_dependencies():
    """Install required packages"""
    print("\n[1/5] Installing dependencies...")
    
    try:
        from unsloth import FastLanguageModel
        print("  ✅ unsloth already installed")
    except ImportError:
        print("  Installing unsloth...")
        os.system("pip install unsloth -q")
        from unsloth import FastLanguageModel
    
    try:
        from trl import SFTTrainer
        print("  ✅ trl already installed")
    except ImportError:
        print("  Installing trl...")
        os.system("pip install trl -q")
    
    try:
        import transformers
        print("  ✅ transformers already installed")
    except ImportError:
        print("  Installing transformers...")
        os.system("pip install transformers -q")
    
    try:
        import datasets
        print("  ✅ datasets already installed")
    except ImportError:
        print("  Installing datasets...")
        os.system("pip install datasets -q")
    
    print("  ✅ Dependencies ready")


def load_and_prepare_model(config: Config):
    """Load Qwen model with LoRA"""
    print(f"\n[2/5] Loading {config.model_name}...")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,  # We have 96GB, no need for 4-bit!
        dtype=torch.bfloat16,  # 16-bit for better precision
    )
    
    print(f"  ✅ Model loaded: {config.model_name}")
    
    # Register special tokens BEFORE training
    print("\n[3/5] Registering special GB tokens...")
    special_tokens = [
        "<GB-ATTACK>", "</GB-ATTACK>",
        "<GB-NORMATIVE>", "</GB-NORMATIVE>",
        "<GB-SEX>", "</GB-SEX>"
    ]
    
    # Add tokens to tokenizer
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)
    print(f"  Added {num_added} special tokens")
    
    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    print(f"  ✅ Tokenizer size: {len(tokenizer)}")
    
    # Apply LoRA
    print("\n[4/5] Applying LoRA adaptation...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Print trainable params
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_training_data(config: Config):
    """Load training and validation datasets in ChatML format"""
    print("\n[5/5] Loading datasets...")
    
    from datasets import Dataset
    import json
    
    def load_jsonl(file_path):
        """Load JSONL file with ChatML formatted text"""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Extract the 'text' field which contains ChatML formatted instruction-input-output
                    if 'text' in item:
                        texts.append(item['text'])
                    else:
                        # Fallback: if no 'text' field, try to reconstruct
                        texts.append(item)
        return texts
    
    train_texts = load_jsonl(config.train_file)
    val_texts = load_jsonl(config.val_file)
    
    # Convert to Dataset format for HF
    train_dataset = Dataset.from_dict({
        "text": train_texts
    })
    
    val_dataset = Dataset.from_dict({
        "text": val_texts
    })
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def setup_trainer(model, tokenizer, train_dataset, val_dataset, config: Config):
    """Setup SFTTrainer for training on responses only"""
    print("\nSetting up trainer...")
    
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=0.01,
        optim="adamw_8bit",  # More memory efficient
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        
        # Logging
        logging_steps=10,
        log_level="info",
        
        # Hardware
        fp16=False,  # We use bfloat16
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        # Reproducibility
        seed=42,
        optim_target_modules=["q_proj", "v_proj"],
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=config.max_seq_length,
        packing=False,  # No packing for span detection accuracy
        peft_config=model.peft_config,
        dataset_kwargs={"add_special_tokens": False}
    )
    
    return trainer


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    config = Config()
    
    print(f"\nConfig Summary:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA Rank: {config.lora_rank}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Output: {config.output_dir}")
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2-5: Load model and data
    model, tokenizer = load_and_prepare_model(config)
    train_dataset, val_dataset = load_training_data(config)
    
    # Step 6: Setup and run trainer
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset, config)
    
    # Train
    train_result = trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SAVING MODEL")
    print("=" * 80)
    
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"\n✅ Model saved to: {config.output_dir}")
    print(f"\nTraining Results:")
    print(f"  Train Loss: {train_result.training_loss:.4f}")
    print(f"\nNext step: Run inference with:")
    print(f"  python inference_qwen_span.py --model-path {config.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
