#!/usr/bin/env python3
"""
Memory-optimized training configuration for H100 when vLLM needs to be stopped
Uses gradient accumulation, gradient checkpointing, and optimized settings
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class H100OptimizedConfig:
    """Optimized configuration for H100 with memory constraints."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-2-7b-chat"
    dtype: str = "bfloat16"  # H100 native support
    use_flash_attention: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True  # Reduces memory by ~30%
    gradient_accumulation_steps: int = 4  # Effective batch size = 4 * per_device_batch_size
    per_device_train_batch_size: int = 2  # Small batch size
    per_device_eval_batch_size: int = 4   # Eval can use larger batch
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj"  # More modules for better results
    
    # Training parameters
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimization
    optim: str = "paged_adamw_32bit"  # Memory-efficient optimizer
    max_grad_norm: float = 1.0
    
    # Sequence length
    max_seq_length: int = 2048
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3  # Keep only 3 checkpoints
    
    # Logging
    logging_steps: int = 10
    log_level: str = "info"
    
    # Data
    train_data_path: str = "services/lora_finetuning/training_data/train.jsonl"
    validation_data_path: str = "services/lora_finetuning/training_data/val.jsonl"
    output_dir: str = "lora_checkpoints/gender_bias_h100"
    
    # DeepSpeed (for multi-GPU, disable if single GPU)
    use_deepspeed: bool = False
    deepspeed_config_path: Optional[str] = None
    
    # Preprocessing
    preprocessing_num_workers: int = 8
    overwrite_cache: bool = False
    
    # Mixed precision
    fp16: bool = False  # Use bfloat16 instead
    bf16: bool = True   # Enabled for H100
    
    # Other settings
    seed: int = 42
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "tensorboard,wandb"
    
    def to_transformers_args(self):
        """Convert to HuggingFace TrainingArguments format."""
        return {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "optim": self.optim,
            "max_grad_norm": self.max_grad_norm,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "seed": self.seed,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "dataloader_num_workers": self.dataloader_num_workers,
            "remove_unused_columns": self.remove_unused_columns,
            "report_to": self.report_to.split(","),
        }
    
    def to_json(self, path: str):
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Preset configurations for different scenarios

class H100Presets:
    """Presets for different H100 scenarios."""
    
    @staticmethod
    def memory_constrained() -> H100OptimizedConfig:
        """Maximum memory optimization - slowest training."""
        config = H100OptimizedConfig()
        config.per_device_train_batch_size = 1
        config.gradient_accumulation_steps = 8
        config.gradient_checkpointing = True
        config.max_seq_length = 1024
        config.eval_steps = 50
        config.save_steps = 50
        config.logging_steps = 5
        return config
    
    @staticmethod
    def balanced() -> H100OptimizedConfig:
        """Balanced between memory and speed - recommended."""
        config = H100OptimizedConfig()
        config.per_device_train_batch_size = 2
        config.gradient_accumulation_steps = 4
        config.max_seq_length = 2048
        return config
    
    @staticmethod
    def high_memory() -> H100OptimizedConfig:
        """For full free H100 - fastest training."""
        config = H100OptimizedConfig()
        config.per_device_train_batch_size = 4
        config.gradient_accumulation_steps = 2
        config.max_seq_length = 2048
        return config


if __name__ == "__main__":
    # Save example configs
    presets = {
        "memory_constrained": H100Presets.memory_constrained(),
        "balanced": H100Presets.balanced(),
        "high_memory": H100Presets.high_memory(),
    }
    
    output_dir = Path("lora_checkpoints/configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, config in presets.items():
        path = output_dir / f"config_{name}.json"
        config.to_json(str(path))
        print(f"✓ Saved {name} config to {path}")
