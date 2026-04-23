#!/usr/bin/env python3
"""
Train an unconditional LM with LoRA on radiology inputs (ReXGradient-160K).

Each training example is the user-turn text sent to MedGemma:
  [Study: {study_description}]  [Indication: {indication}]  Findings:\n{findings}

Loss is computed on all non-padding tokens (unconditional generation).

Usage:
    python f2i_input_modeling.py \
        --output checkpoints/f2i_input_model \
        --epochs 3 \
        --batch-size 4
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B"
LLAMA_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
INPUT_START_TOKEN = "<|INPUT_START|>"
INPUT_END_TOKEN = "<|INPUT_END|>"


def get_model_and_tokenizer(model_name: str, use_4bit: bool, use_flash_attention: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if use_flash_attention and torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


class GenerationLoggingCallback(TrainerCallback):
    """At each eval step, generate completions from a fixed set of prompts."""

    def __init__(
        self,
        tokenizer,
        eval_dataset,
        num_examples: int = 3,
        prompt_tokens: int = 5,
        max_new_tokens: int = 200,
        use_wandb: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.use_wandb = use_wandb

        indices = list(range(min(num_examples, len(eval_dataset))))
        self.prompt_ids = []
        for idx in indices:
            ids = eval_dataset[idx]["input_ids"]
            non_pad = [t for t in ids if t != tokenizer.pad_token_id]
            self.prompt_ids.append(non_pad[: prompt_tokens])

    def on_evaluate(self, _args: TrainingArguments, state: TrainerState,
                    _control: TrainerControl, model=None, **_kwargs):
        if model is None:
            return

        model.eval()
        device = next(model.parameters()).device
        rows = []

        for i, prompt in enumerate(self.prompt_ids):
            input_ids = torch.tensor([prompt], device=device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_text = self.tokenizer.decode(prompt, skip_special_tokens=False)
            generated_text = self.tokenizer.decode(
                output_ids[0][len(prompt):], skip_special_tokens=False
            )
            if INPUT_END_TOKEN in generated_text:
                generated_text = generated_text.split(INPUT_END_TOKEN)[0]
            rows.append((prompt_text, generated_text))
            print(f"\n--- Generation example {i + 1} (step {state.global_step}) ---")
            print(f"[PROMPT]    {prompt_text}")
            print(f"[GENERATED] {generated_text}")

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["prompt", "generated"])
                    for prompt_text, generated_text in rows:
                        table.add_data(prompt_text, generated_text)
                    wandb.log({"generation_examples": table}, step=state.global_step)
            except ImportError:
                pass


def _str_or_none(val) -> str | None:
    return val if isinstance(val, str) and val.strip() else None


def build_conditioning_prefix(
    sex: str | None,
    age: str | None,
    study_description: str | None,
) -> str:
    """Build the conditioning prefix (no loss computed on these tokens).

    Format: "Sex: {sex}  Age: {age}  Study: {study_description}"
    Only non-empty fields are included.
    """
    parts = []
    if sex:
        parts.append(f"Sex: {sex}")
    if age:
        parts.append(f"Age: {age}")
    if study_description:
        parts.append(f"Study: {study_description}")
    return "  ".join(parts)


def build_input_text(findings: str, indication: str | None) -> str:
    """Construct the generated portion of each example (loss is computed here)."""
    parts = []
    if indication:
        parts.append(f"Indication: {indication}")
    parts.append(f"Findings:\n{findings}")
    return "\n\n".join(parts)


def _detect_col(col_names: list[str], keywords: list[str]) -> str | None:
    return next((c for c in col_names if any(k in c.lower() for k in keywords)), None)


def build_prefix_and_full_texts(examples) -> tuple[list[str], list[str]]:
    """Return (prefix_texts, full_texts) for each example.

    prefix_text  — conditioning context, tokenized separately to determine mask boundary
    full_text    — prefix + <|INPUT_START|> generated_content <|INPUT_END|>
    """
    prefixes, fulls = [], []
    for findings, indication, study_desc, sex, age in zip(
        examples["_findings"],
        examples["_indication"],
        examples["_study_description"],
        examples["_sex"],
        examples["_age"],
    ):
        prefix = build_conditioning_prefix(
            sex=_str_or_none(sex),
            age=_str_or_none(age),
            study_description=_str_or_none(study_desc),
        )
        content = build_input_text(
            findings=findings,
            indication=_str_or_none(indication),
        )
        generated = f"{INPUT_START_TOKEN} {content} {INPUT_END_TOKEN}"
        full = (prefix + "  " + generated).strip() if prefix else generated
        prefixes.append(prefix)
        fulls.append(full)
    return prefixes, fulls


def tokenize(examples, tokenizer, max_length: int) -> dict:
    eos = tokenizer.eos_token or ""
    prefix_texts, full_texts = build_prefix_and_full_texts(examples)
    full_texts = [(t + " " + eos).strip() for t in full_texts]

    enc = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
        add_special_tokens=True,
    )

    # Tokenize prefixes without padding to measure how many tokens to mask.
    # Use add_special_tokens=False so the BOS token is not double-counted.
    prefix_enc = tokenizer(
        prefix_texts,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    # +1 to also mask the BOS token prepended by the full-text tokenization.
    prefix_lengths = [len(ids) + 1 for ids in prefix_enc["input_ids"]]

    labels = []
    for ids, attn, prefix_len in zip(enc["input_ids"], enc["attention_mask"], prefix_lengths):
        label_row = [
            tok if (mask and pos >= prefix_len) else -100
            for pos, (tok, mask) in enumerate(zip(ids, attn))
        ]
        labels.append(label_row)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def prepare_dataset(split: str, num_samples: int):
    """Load ReXGradient and normalize column names.

    Renames dataset columns to canonical internal names regardless of the
    dataset's actual column naming. Any optional column not found is added
    as all-None. No value normalization is performed — values are used as-is.

    Canonical columns:
      _findings          — required
      _indication        — optional, included in generated portion
      _study_description — optional, conditioning prefix
      _sex               — optional, conditioning prefix
      _age               — optional, conditioning prefix
    """
    from datasets import load_dataset
    ds = load_dataset("rajpurkarlab/ReXGradient-160K")[split]

    col_names = ds.column_names
    findings_col   = _detect_col(col_names, ["finding"])
    indication_col = _detect_col(col_names, ["indication", "clinical_info", "history"])
    study_desc_col = _detect_col(col_names, ["study_description", "study description", "modality", "procedure"])
    sex_col        = _detect_col(col_names, ["sex", "gender"])
    age_col        = _detect_col(col_names, ["age"])

    if findings_col is None:
        raise ValueError(f"Could not find a findings column in {col_names}")

    print(f"  [{split}] findings='{findings_col}', indication='{indication_col}', "
          f"study_description='{study_desc_col}', sex='{sex_col}', age='{age_col}'  "
          f"({len(ds):,} examples)")

    rename_map = {findings_col: "_findings"}
    if indication_col:
        rename_map[indication_col] = "_indication"
    if study_desc_col:
        rename_map[study_desc_col] = "_study_description"
    if sex_col:
        rename_map[sex_col] = "_sex"
    if age_col:
        rename_map[age_col] = "_age"

    ds = ds.rename_columns(rename_map)

    for col in ["_indication", "_study_description", "_sex", "_age"]:
        if col not in ds.column_names:
            ds = ds.add_column(col, [None] * len(ds))

    ds = ds.select_columns(["_findings", "_indication", "_study_description", "_sex", "_age"])

    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Train an unconditional LM on radiology inputs (ReXGradient-160K) with LoRA.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and final adapter (default: ./f2i_input_model)",
    )
    parser.add_argument(
        "--load-adapter",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to an existing LoRA checkpoint to load and continue training.",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=-1,
        help="-1 = all training samples",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=250,
        help="Max examples to use for evaluation (default: 250)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size (default: 4)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Peak learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use bf16; needs more VRAM)",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Use Flash Attention 2 (requires flash-attn installed)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="f2i-input-modeling",
        help="W&B project name (default: f2i-input-modeling)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument(
        "--gen-examples",
        type=int,
        default=3,
        metavar="N",
        help="Number of examples to generate at each eval step (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--gen-prompt-tokens",
        type=int,
        default=5,
        metavar="N",
        help="Number of tokens to use as the generation prompt (default: 5)",
    )
    parser.add_argument(
        "--gen-max-new-tokens",
        type=int,
        default=300,
        metavar="N",
        help="Max new tokens to generate per example (default: 300)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "f2i_input_model"
    args.output_dir = (
        Path(args.output_dir).parent
        / f"{Path(args.output_dir).name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)

    print("Loading dataset...")
    train_ds = prepare_dataset("train", args.num_train_samples)
    eval_ds = prepare_dataset("test", -1)
    if args.max_eval_samples > 0 and args.max_eval_samples < len(eval_ds):
        eval_ds = eval_ds.shuffle(seed=42).select(range(args.max_eval_samples))
    print(f"Train: {len(train_ds):,} examples  |  Eval: {len(eval_ds):,} examples")

    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(
        args.model,
        use_4bit=not args.no_4bit,
        use_flash_attention=args.flash_attention,
    )

    load_adapter_path = Path(args.load_adapter).resolve() if args.load_adapter else None
    if load_adapter_path is not None:
        if not load_adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter path not found: {load_adapter_path}")
        if not args.no_4bit and torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)
        print(f"Loading existing LoRA adapter from {load_adapter_path}...")
        model = PeftModel.from_pretrained(model, str(load_adapter_path), is_trainable=True)
    else:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=LLAMA_LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tok_fn = lambda ex: tokenize(ex, tokenizer, args.max_length)

    print("Tokenizing...")
    tokenized_train = train_ds.map(
        tok_fn,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=4,
        desc="Tokenize train",
    )
    tokenized_eval = eval_ds.map(
        tok_fn,
        batched=True,
        remove_columns=eval_ds.column_names,
        num_proc=4,
        desc="Tokenize eval",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    training_args_kw = {"run_name": args.wandb_run_name} if args.wandb_run_name else {}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if use_wandb else "none",
        dataloader_num_workers=4,
        **training_args_kw,
    )

    callbacks = []
    if args.gen_examples > 0:
        callbacks.append(GenerationLoggingCallback(
            tokenizer=tokenizer,
            eval_dataset=tokenized_eval,
            num_examples=args.gen_examples,
            prompt_tokens=args.gen_prompt_tokens,
            max_new_tokens=args.gen_max_new_tokens,
            use_wandb=use_wandb,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
