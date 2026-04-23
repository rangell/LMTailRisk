"""
Fine-tune MedGemma-4B-IT to produce CRIMSON evaluation scores.

Input:  CRIMSON prompt (excl. CONTEXT_GUIDELINES) with GT + candidate reports
Output: raw_evaluation JSON

Supports multi-GPU via Accelerate / FSDP and uses LoRA (r=16) on the
language model with bf16 mixed precision.

Usage (single node, launched via accelerate):
    accelerate launch finetune_medgemma.py \
        --train_jsonl ../data/finetuned_medgemma/train_data.jsonl \
        --output_dir ../data/finetuned_medgemma/checkpoints \
        --model_id google/medgemma-4b-it \
        --num_samples 300
"""

import argparse
import json
import os
import random
import traceback

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from dataset import CRIMSONDataset, collate_fn


# ------------------------------------------------------------------ helpers
def print_main(msg: str):
    """Print only on rank 0."""
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        print(msg)


def load_jsonl(
    path: str,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load a JSONL file, optionally shuffling and limiting the number of samples.

    With shuffle=True, all valid entries are loaded, shuffled with a fixed seed,
    then truncated to max_samples.  This avoids bias from sequential ordering.
    """
    data = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("raw_evaluation") is not None:
                data.append(entry)
            if not shuffle and max_samples is not None and len(data) >= max_samples:
                break

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)

    if max_samples is not None:
        data = data[:max_samples]

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma-4B for CRIMSON evaluation scoring"
    )

    # Data
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to train_data.jsonl",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to use from the JSONL (-1 = all, default: -1)",
    )

    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/medgemma-4b-it",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for model weights",
    )

    # Training hyper-parameters
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4500)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["epoch", "steps", "no"],
                        help="When to save checkpoints (default: epoch)")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save every N steps (only used if save_strategy=steps)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max checkpoints to keep on disk (default: 3)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                        help="Num dataloader workers per GPU (default: 2)")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    max_samples = None if args.num_samples < 0 else args.num_samples
    print_main(f"Loading data from {args.train_jsonl} "
               f"({'all' if max_samples is None else max_samples} samples, shuffled) ...")
    data = load_jsonl(args.train_jsonl, max_samples=max_samples, shuffle=True, seed=args.seed)
    print_main(f"  Loaded {len(data)} valid samples")

    print_main(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print_main("Building dataset ...")
    dataset = CRIMSONDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print_main(f"  Dataset size: {len(dataset)}")

    print_main(f"Loading model: {args.model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )

    print_main("Applying LoRA ...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print_main(f"Starting training for {args.num_epochs} epochs ...")
    try:
        trainer.train()
    except Exception:
        traceback.print_exc()
        raise

    final_dir = os.path.join(args.output_dir, "final")
    print_main(f"Saving final model to {final_dir} ...")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print_main("Done!")


if __name__ == "__main__":
    main()
