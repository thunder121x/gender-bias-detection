#!/usr/bin/env python3
"""
Unsloth LoRA fine-tuning for Thai Gender Bias Span Detection.

This keeps the original system prompt from assets/*.jsonl, normalizes assistant
answers to compact JSON label arrays, and trains loss only on the final
assistant response.
"""

import argparse
import json
import logging

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

LABELS = ("GB-NORMATIVE", "GB-SEX", "GB-ATTACK")
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def compact_json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def extract_user_text(content):
    marker = "Input:"
    if marker in content:
        return content.rsplit(marker, 1)[-1].strip()
    return content.strip()


def normalize_assistant_content(content):
    empty_result = {label: [] for label in LABELS}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return compact_json(empty_result)

    if isinstance(parsed, dict) and all(label in parsed for label in LABELS):
        normalized = {
            label: [span for span in parsed.get(label, []) if isinstance(span, str)]
            for label in LABELS
        }
        return compact_json(normalized)

    normalized = {label: [] for label in LABELS}
    for span in parsed.get("spans", []) if isinstance(parsed, dict) else []:
        if not isinstance(span, dict):
            continue
        label = span.get("label")
        text = span.get("text")
        if label in normalized and isinstance(text, str) and text:
            normalized[label].append(text)

    return compact_json(normalized)


def normalize_messages(messages):
    normalized = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            normalized.append({"role": "system", "content": content})
        elif role == "user":
            normalized.append({"role": "user", "content": extract_user_text(content)})
        elif role == "assistant":
            normalized.append({"role": "assistant", "content": normalize_assistant_content(content)})
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def extract_input_ids(tokenized_chat):
    if hasattr(tokenized_chat, "ids"):
        return tokenized_chat.ids
    if isinstance(tokenized_chat, dict):
        input_ids = tokenized_chat["input_ids"]
        return input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
    if hasattr(tokenized_chat, "input_ids"):
        return tokenized_chat.input_ids
    return tokenized_chat


def resolve_text_tokenizer(processing_class):
    if hasattr(processing_class, "encode"):
        return processing_class
    if hasattr(processing_class, "tokenizer") and hasattr(processing_class.tokenizer, "encode"):
        return processing_class.tokenizer
    raise TypeError(
        f"{type(processing_class).__name__} does not expose encode() or a text tokenizer"
    )


def ensure_chat_template(tokenizer, processing_class=None):
    if getattr(tokenizer, "chat_template", None):
        return

    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    if processing_class is not None and hasattr(processing_class, "chat_template"):
        processing_class.chat_template = QWEN_CHAT_TEMPLATE
    logger.info("Tokenizer has no chat_template; using Qwen ChatML fallback")


def preprocess_function(examples, tokenizer, max_length):
    all_input_ids = []
    all_labels = []
    assistant_start_token = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    for messages in examples["messages"]:
        messages = normalize_messages(messages)
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
        )

        input_ids = extract_input_ids(tokenized_chat)
        labels = list(input_ids)

        start_positions = []
        for i in range(len(input_ids) - len(assistant_start_token) + 1):
            if input_ids[i:i + len(assistant_start_token)] == assistant_start_token:
                start_positions.append(i)

        if start_positions:
            content_start_idx = start_positions[-1] + len(assistant_start_token)
            labels[:content_start_idx] = [-100] * content_start_idx
        else:
            labels = [-100] * len(input_ids)

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


def resolve_report_to(requested):
    if requested == "none":
        return "none"
    if requested != "auto":
        return requested

    try:
        import tensorboard  # noqa: F401
        return "tensorboard"
    except ImportError:
        try:
            import tensorboardX  # noqa: F401
            return "tensorboard"
        except ImportError:
            logger.info("TensorBoard is not installed; disabling trainer reporting")
            return "none"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 with Unsloth LoRA")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--model", type=str, default="unsloth/Qwen3.5-2B-Base")
    parser.add_argument("--output-dir", type=str, default="lora_checkpoints/gender_bias_qwen35_2b_unsloth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA instead of bf16 LoRA.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit loading instead of bf16 LoRA.")
    parser.add_argument("--report-to", type=str, default="auto")
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")

    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, processing_class = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        load_in_16bit=not args.load_in_4bit and not args.load_in_8bit,
        full_finetuning=False,
    )
    tokenizer = resolve_text_tokenizer(processing_class)
    ensure_chat_template(tokenizer, processing_class)
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=args.max_seq_length,
    )

    train_raw = load_jsonl(args.train_file)
    val_raw = load_jsonl(args.val_file)

    train_dataset = train_raw.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_raw.column_names,
    )
    val_dataset = val_raw.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=val_raw.column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        tf32=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=resolve_report_to(args.report_to),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )

    logger.info("Starting Unsloth training with model: %s", args.model)
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    if processing_class is not tokenizer:
        processing_class.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done! LoRA adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
