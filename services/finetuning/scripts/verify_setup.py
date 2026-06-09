#!/usr/bin/env python3
"""
Verify environment and setup for H100 fine-tuning
Checks CUDA, PyTorch, dependencies, and data
"""

import sys
import subprocess
import json
from pathlib import Path


def check_cuda():
    """Check CUDA availability and version."""
    print("\n🔹 CUDA CHECK:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print(f"  ✗ CUDA not available")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_pytorch():
    """Check PyTorch and related packages."""
    print("\n🔹 PYTORCH CHECK:")
    try:
        import torch
        print(f"  ✓ torch: {torch.__version__}")
        
        # Check bfloat16 support
        if hasattr(torch.cuda, 'is_bf16_supported'):
            bf16_support = torch.cuda.is_bf16_supported()
            print(f"  ✓ bfloat16 support: {bf16_support}")
        
        # Check Flash Attention
        try:
            from torch.nn.attention import sdpa_kernel
            print(f"  ✓ Flash Attention 2 available")
        except:
            print(f"  ⚠ Flash Attention 2 not available")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_transformers():
    """Check transformers and related packages."""
    print("\n🔹 TRANSFORMERS CHECK:")
    try:
        import transformers
        print(f"  ✓ transformers: {transformers.__version__}")
        
        from peft import __version__ as peft_version
        print(f"  ✓ peft: {peft_version}")
        
        import datasets
        print(f"  ✓ datasets: {datasets.__version__}")
        
        import bitsandbytes
        print(f"  ✓ bitsandbytes: {bitsandbytes.__version__}")
        
        return True
    except ImportError as e:
        print(f"  ✗ Missing package: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_data():
    """Check training data files."""
    print("\n🔹 DATA CHECK:")
    base_path = Path("services/lora_finetuning/training_data")
    
    files_ok = True
    for filename in ["train.jsonl", "val.jsonl"]:
        filepath = base_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            lines = sum(1 for _ in open(filepath))
            print(f"  ✓ {filename}: {lines:,} lines ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename}: NOT FOUND")
            files_ok = False
    
    return files_ok


def check_scripts():
    """Check training scripts exist."""
    print("\n🔹 SCRIPTS CHECK:")
    scripts = [
        "train_simple.py",
        "train_lora_h100.py",
        "h100_config.py",
        "inspect_data.py",
        "merge_lora.py",
    ]
    
    all_ok = True
    for script in scripts:
        if Path(script).exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script}: NOT FOUND")
            all_ok = False
    
    return all_ok


def check_gpu_memory():
    """Check available GPU memory."""
    print("\n🔹 GPU MEMORY CHECK:")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], 
                              capture_output=True, text=True)
        free_memory_mb = int(result.stdout.strip())
        free_memory_gb = free_memory_mb / 1024
        
        if free_memory_gb > 80:
            print(f"  ✓ Free VRAM: {free_memory_gb:.1f} GB (EXCELLENT)")
            print(f"    → Can use 'high_memory' preset")
        elif free_memory_gb > 60:
            print(f"  ✓ Free VRAM: {free_memory_gb:.1f} GB (GOOD)")
            print(f"    → Can use 'balanced' preset")
        elif free_memory_gb > 40:
            print(f"  ⚠ Free VRAM: {free_memory_gb:.1f} GB (LIMITED)")
            print(f"    → Use 'memory_constrained' preset")
        else:
            print(f"  ✗ Free VRAM: {free_memory_gb:.1f} GB (NOT ENOUGH)")
            print(f"    → Stop vLLM and Triton first")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False


def check_hf_cache():
    """Check HuggingFace cache."""
    print("\n🔹 HUGGINGFACE CACHE CHECK:")
    try:
        import os
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hf_path = Path(hf_home)
        
        if hf_path.exists():
            size_gb = sum(f.stat().st_size for f in hf_path.rglob("*") if f.is_file()) / (1024**3)
            print(f"  ✓ HF cache: {hf_path}")
            print(f"  ✓ Cache size: {size_gb:.2f} GB")
        else:
            print(f"  ⚠ HF cache not yet created")
        
        return True
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return False


def main():
    print("=" * 80)
    print("H100 FINE-TUNING ENVIRONMENT VERIFICATION")
    print("=" * 80)
    
    checks = [
        ("CUDA", check_cuda),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("Data", check_data),
        ("Scripts", check_scripts),
        ("GPU Memory", check_gpu_memory),
        ("HF Cache", check_hf_cache),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    
    all_ok = all(results.values())
    
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 80)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
        print("\nNext steps:")
        print("  1. Run: python3 train_simple.py --preset balanced")
        print("  2. Monitor: watch -n 1 nvidia-smi")
        print("  3. View logs: tensorboard --logdir lora_checkpoints/gender_bias_h100")
    else:
        print("⚠️  SOME CHECKS FAILED - READ ERRORS ABOVE")
        if not results["GPU Memory"]:
            print("\nQuick fix: pkill -f 'VLLM::EngineCore' && pkill -f tritonserver")
        if not results["Transformers"]:
            print("\nQuick fix: pip install -q transformers peft datasets bitsandbytes")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
