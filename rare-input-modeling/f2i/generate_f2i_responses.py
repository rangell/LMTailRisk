"""
Use MedGemma (google/medgemma-4b-it) to generate impressions from radiology findings.

Loads the ReXGradient-160K dataset via rad_data.py, runs batch inference,
and saves results to a JSONL file.

Usage:
    python generate_responses.py \
        --split test \
        --output results/medgemma_f2i.jsonl \
        --num_samples 500 \
        --batch_size 8
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from rad_data import load_rexgradient_dataset

MODEL_ID = "google/medgemma-4b-it"

SYSTEM_PROMPT = (
    "You are an expert radiologist. Given information about a radiology study, "
    "write a concise Impression section that summarizes the key findings and their "
    "clinical significance. Respond with the Impression only — no preamble."
)


def build_prompt(
    findings: str,
    indication: str | None = None,
    study_description: str | None = None,
) -> list[dict]:
    parts = []
    if study_description:
        parts.append(f"Study: {study_description}")
    if indication:
        parts.append(f"Indication: {indication}")
    parts.append(f"Findings:\n{findings}")
    parts.append("Impression:")
    return [
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def load_model(model_id: str, cache_dir: str | None):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_impressions(
    model,
    tokenizer,
    findings_list: list[str],
    batch_size: int,
    max_new_tokens: int,
    indications: list[str | None] | None = None,
    study_descriptions: list[str | None] | None = None,
) -> list[str]:
    impressions = []
    n = len(findings_list)

    def _get(lst, idx):
        return lst[idx] if lst is not None else None

    for i in tqdm(range(0, n, batch_size), desc="Generating", unit="batch"):
        batch_end = min(i + batch_size, n)
        # print(f"  Batch {i // batch_size + 1} / {(n + batch_size - 1) // batch_size}")

        # Apply chat template for each item
        encoded_batch = []
        for j in range(i, batch_end):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + build_prompt(
                findings=findings_list[j],
                indication=_get(indications, j),
                study_description=_get(study_descriptions, j),
            )
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded_batch.append(text)

        inputs = tokenizer(
            encoded_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        for out in output_ids:
            generated = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            impressions.append(generated.strip())

    return impressions


def generate_impressions_vllm(
    model_id: str,
    findings_list: list[str],
    batch_size: int,
    max_new_tokens: int,
    indications: list[str | None] | None = None,
    study_descriptions: list[str | None] | None = None,
    cache_dir: str | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    enforce_eager: bool = False,
    temperature: float = 0.0,
) -> list[str]:
    from vllm import LLM, SamplingParams

    def _get(lst, idx):
        return lst[idx] if lst is not None else None

    print(f"Loading vLLM model: {model_id} (tp={tensor_parallel_size})")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        download_dir=cache_dir,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    impressions = []
    n = len(findings_list)

    for i in tqdm(range(0, n, batch_size), desc="Generating", unit="batch"):
        batch_end = min(i + batch_size, n)
        messages_batch = [
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + build_prompt(
                findings=findings_list[j],
                indication=_get(indications, j),
                study_description=_get(study_descriptions, j),
            )
            for j in range(i, batch_end)
        ]
        outputs = llm.chat(messages_batch, sampling_params, use_tqdm=False)
        impressions.extend(out.outputs[0].text.strip() for out in outputs)

    return impressions


def main():
    parser = argparse.ArgumentParser(description="Run MedGemma F2I inference on ReXGradient")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--output", type=str, default="results/medgemma_f2i.jsonl")
    parser.add_argument("--num_samples", type=int, default=-1, help="-1 = all samples")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"])
    # vLLM-specific
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    # chunked parallel jobs
    parser.add_argument("--job_index", type=int, default=None,
                        help="0-based index of this job (use with --num_jobs)")
    parser.add_argument("--num_jobs", type=int, default=None,
                        help="Total number of parallel jobs (use with --job_index)")
    args = parser.parse_args()

    if (args.job_index is None) != (args.num_jobs is None):
        parser.error("--job_index and --num_jobs must be provided together")
    if args.job_index is not None and not (0 <= args.job_index < args.num_jobs):
        parser.error(f"--job_index must be in [0, num_jobs-1], got {args.job_index} with --num_jobs {args.num_jobs}")

    # Load dataset
    print(f"Loading ReXGradient-160K split='{args.split}' ...")
    ds = load_rexgradient_dataset(split=args.split)
    print(f"  Total samples: {len(ds)}")

    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))
        print(f"  Using {len(ds)} samples")

    # Identify columns
    col_names = ds.column_names
    print(f"  Columns: {col_names}")

    def _col(keywords: list[str]) -> str | None:
        return next((c for c in col_names if any(k in c.lower() for k in keywords)), None)

    findings_col = _col(["finding"])
    impression_col = _col(["impression"])
    indication_col = _col(["indication", "clinical_info", "history"])
    study_desc_col = _col(["study_description", "study description", "modality", "procedure"])

    if findings_col is None:
        raise ValueError(f"Could not find a 'findings' column in {col_names}")
    print(f"  findings='{findings_col}', impression='{impression_col}', "
          f"indication='{indication_col}', study_description='{study_desc_col}'")

    def _extract(col: str | None) -> list[str | None] | None:
        if col is None:
            return None
        vals = ds[col]
        return [v if isinstance(v, str) and v.strip() else None for v in vals]

    findings_list = list(ds[findings_col])
    indications = _extract(indication_col)
    study_descriptions = _extract(study_desc_col)
    impression_list = list(ds[impression_col]) if impression_col is not None else None

    # ------------------------------------------------------------------
    # Slice for this job
    # ------------------------------------------------------------------
    global_indices = list(range(len(findings_list)))
    if args.job_index is not None:
        n_total = len(findings_list)
        chunk_size = (n_total + args.num_jobs - 1) // args.num_jobs
        start = args.job_index * chunk_size
        end = min(start + chunk_size, n_total)
        print(f"  Job {args.job_index}/{args.num_jobs}: rows {start}–{end - 1} of {n_total}")
        global_indices  = global_indices[start:end]
        findings_list   = findings_list[start:end]
        indications     = indications[start:end]     if indications     is not None else None
        study_descriptions = study_descriptions[start:end] if study_descriptions is not None else None
        impression_list = impression_list[start:end] if impression_list is not None else None

    # Resolve output path (auto-name when chunked and no explicit --output)
    if args.job_index is not None and args.output == "results/medgemma_f2i.jsonl":
        out_path = f"results/medgemma_f2i_job{args.job_index:04d}_of{args.num_jobs:04d}.jsonl"
    else:
        out_path = args.output

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    # Run inference
    print(f"Generating impressions via {args.backend} (batch_size={args.batch_size}) ...")
    if args.backend == "vllm":
        impressions = generate_impressions_vllm(
            model_id=args.model_id,
            findings_list=findings_list,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            indications=indications,
            study_descriptions=study_descriptions,
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            temperature=args.temperature,
        )
    else:
        model, tokenizer = load_model(args.model_id, args.cache_dir)
        impressions = generate_impressions(
            model, tokenizer, findings_list, args.batch_size, args.max_new_tokens,
            indications=indications,
            study_descriptions=study_descriptions,
        )
    
    with open(out_path, "w") as f:
        for local_idx, (global_idx, findings, pred) in enumerate(
            zip(global_indices, findings_list, impressions)
        ):
            record = {
                "id": global_idx,
                "findings": findings,
                "predicted_impression": pred,
            }
            if impression_list is not None:
                record["reference_impression"] = impression_list[local_idx]
            if indications is not None:
                record["indication"] = indications[local_idx]
            if study_descriptions is not None:
                record["study_description"] = study_descriptions[local_idx]
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(impressions)} records to {out_path}")


if __name__ == "__main__":
    main()