> **Note:** scripts referenced in this guide (train_simple.py, h100_config.py, merge utilities) now live in `services/finetuning/scripts/` and `services/lora_finetuning/`.

# H100 Fine-tuning Guide for Gender Bias Detection

## Quick Start

### Step 1: Stop vLLM to Free Memory

```bash
# Check running processes
nvidia-smi

# Kill vLLM (adjust PID as needed)
kill 2090231  # VLLM::EngineCore PID
kill 3551605  # python3 PID (if running your inference)

# Stop Triton if you want more space
pkill -f tritonserver

# Verify GPU is free
nvidia-smi
```

Expected after stopping vLLM: ~95GB free VRAM on H100

### Step 2: Install Dependencies

```bash
pip install -q \
    transformers==4.40.2 \
    torch==2.2.0 \
    peft==0.10.0 \
    datasets==2.18.0 \
    bitsandbytes==0.42.0 \
    tensorboard==2.15.1 \
    wandb==0.16.3

# Verify CUDA and bfloat16 support
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'bfloat16: {torch.cuda.is_bf16_supported()}')"
```

### Step 3: Choose Training Preset

```bash
# Memory-constrained (max compatibility, slowest)
python3 train_simple.py --preset memory_constrained

# Balanced (recommended - good speed/memory tradeoff)
python3 train_simple.py --preset balanced

# High memory (fastest, needs full free H100)
python3 train_simple.py --preset high_memory
```

### Step 4: Monitor Training

```bash
# In another terminal, monitor GPU
watch -n 1 nvidia-smi

# Or view TensorBoard
tensorboard --logdir lora_checkpoints/gender_bias_h100
```

---

## Configuration Presets

### Memory Constrained
- **Best for**: When you need to keep other processes running
- **Batch size**: 1
- **Accumulation steps**: 8 (effective batch: 8)
- **Max sequence**: 1024 tokens
- **Memory usage**: ~40-50 GB
- **Training speed**: Slowest
- **Use case**: When you can't stop everything

```bash
python3 train_simple.py --preset memory_constrained
```

### Balanced (Recommended)
- **Best for**: Standard use case
- **Batch size**: 2
- **Accumulation steps**: 4 (effective batch: 8)
- **Max sequence**: 2048 tokens
- **Memory usage**: ~60-70 GB
- **Training speed**: Moderate
- **Use case**: Best balance of speed and stability

```bash
python3 train_simple.py --preset balanced
```

### High Memory
- **Best for**: Full H100 available
- **Batch size**: 4
- **Accumulation steps**: 2 (effective batch: 8)
- **Max sequence**: 2048 tokens
- **Memory usage**: ~80-90 GB
- **Training speed**: Fastest
- **Use case**: Maximum speed when vLLM and Triton are fully stopped

```bash
python3 train_simple.py --preset high_memory
```

---

## Advanced Usage

### Custom Training Parameters

```bash
python3 train_simple.py \
    --preset balanced \
    --model meta-llama/Llama-2-13b-chat \
    --output-dir my_checkpoints/custom_model \
    --train-file path/to/custom_train.jsonl \
    --val-file path/to/custom_val.jsonl
```

### Using Custom Config File

```python
from h100_config import H100OptimizedConfig

# Load and modify preset
config = H100OptimizedConfig()
config.per_device_train_batch_size = 3
config.num_train_epochs = 2
config.learning_rate = 1e-4

# Save custom config
config.to_json("my_config.json")
```

---

## Expected Performance

### Training Times (Llama-2-7B)

| Preset | Effective Batch | Time per Epoch | Total (3 epochs) |
|--------|-----------------|----------------|------------------|
| memory_constrained | 8 | ~4 hours | ~12 hours |
| balanced | 8 | ~2.5 hours | ~7.5 hours |
| high_memory | 8 | ~2 hours | ~6 hours |

### Memory Usage

| Preset | Peak VRAM | Safe Margin |
|--------|-----------|-------------|
| memory_constrained | 50 GB | 45 GB free |
| balanced | 70 GB | 25 GB free |
| high_memory | 90 GB | 5 GB free |

---

## Troubleshooting

### Out of Memory Error

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Use `memory_constrained` preset
2. Reduce `--max-seq-length` to 1024 or 512
3. Further reduce batch size in config

### Slow Training / Low GPU Utilization

**Causes**:
- Using too small batch size
- Tokenization bottleneck
- Disk I/O issues

**Solutions**:
1. Use `high_memory` preset if space allows
2. Increase `dataloader_num_workers` to 8-16
3. Use faster storage (NVMe)

### CUDA Version Mismatch

**Error**: `ImportError: libcuda.so.13 not found`

**Solution**:
```bash
# Install compatible torch version
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Model Download Issues

**Error**: `Connection timeout when downloading model`

**Solution**:
```bash
# Manually download and cache model
huggingface-cli download meta-llama/Llama-2-7b-chat

# Or use local model path
python3 train_simple.py --model /path/to/local/model
```

---

## Output Structure

After training, your checkpoint directory will contain:

```
lora_checkpoints/gender_bias_h100/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # LoRA weights
├── training_args.bin            # Training arguments
├── training_summary.json        # Training metrics
├── training_config.json         # Your config
├── runs/                        # TensorBoard logs
└── checkpoint-*/               # Intermediate checkpoints
```

---

## Loading and Using Your Fine-tuned Model

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model, 
    "lora_checkpoints/gender_bias_h100"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")

# Use model
input_text = "ผู้ชายต้องเป็นผู้นำเสมอ"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

---

## Performance Optimization Tips

1. **Pre-cache datasets**: Converts JSONL to Arrow format for faster loading
2. **Use DDP**: For multi-GPU training (future versions)
3. **Flash Attention 2**: Already enabled - provides 2-3x speedup
4. **Bfloat16**: Uses native H100 support (no conversion overhead)
5. **Gradient checkpointing**: Saves 30% memory with minimal speed penalty

---

## Next Steps

After training completes:

1. **Evaluate model** on gender bias detection task
2. **Merge LoRA weights** with base model for inference
3. **Deploy** using vLLM or TorchServe
4. **Monitor** performance on production data

---

**Last updated**: 2026-05-03
**Tested on**: NVIDIA H100 NVL (95GB VRAM)
**CUDA Version**: 13.0
