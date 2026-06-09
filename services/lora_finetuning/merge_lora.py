#!/usr/bin/env python3
"""
Utility to merge LoRA weights with base model for inference
Converts fine-tuned LoRA adapter to standalone model
"""

import argparse
import torch
import logging
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
):
    """
    Merge LoRA adapter weights with base model.
    
    Args:
        base_model_path: Path or HF ID of base model
        lora_path: Path to LoRA checkpoint
        output_path: Where to save merged model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_id: HF repo ID if pushing
    """
    
    logger.info("=" * 80)
    logger.info("MERGING LoRA WEIGHTS WITH BASE MODEL")
    logger.info("=" * 80)
    
    # Load base model
    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge
    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Save
    logger.info(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    # Push to hub if requested
    if push_to_hub and hub_repo_id:
        logger.info(f"Pushing to Hub: {hub_repo_id}")
        merged_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
    
    logger.info("=" * 80)
    logger.info("✅ MERGE COMPLETE!")
    logger.info(f"Model saved to: {output_path}")
    logger.info("=" * 80)
    
    return merged_model


def get_model_size(path: str) -> float:
    """Get total size of model in GB."""
    total_size = 0
    for file in Path(path).rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024**3)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-chat",
        help="Base model path or HF ID"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        help="HuggingFace repo ID (required if pushing)"
    )
    
    args = parser.parse_args()
    
    # Verify paths
    if not Path(args.lora_path).exists():
        logger.error(f"LoRA path not found: {args.lora_path}")
        return
    
    # Merge
    merge_lora_weights(
        args.base_model,
        args.lora_path,
        args.output_path,
        args.push_to_hub,
        args.hub_repo_id,
    )
    
    # Print size info
    size = get_model_size(args.output_path)
    logger.info(f"📊 Merged model size: {size:.2f} GB")


if __name__ == "__main__":
    main()
