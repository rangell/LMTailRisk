"""
Generate radiology report classifications using MedGemma via vLLM.

Loads N samples from ReXGradient-160K, runs inference with MedGemma,
and saves results to a JSONL file for manual inspection.

Usage:
    python qwen3_summaries.py
    python qwen3_summaries.py --num_samples 20 --output out.jsonl
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rad_data import load_rexgradient_dataset

# MODEL_ID = "google/medgemma-4b-it"
MODEL_ID = "google/medgemma-4b-it"


PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "f2i_input_summary_prompt.json")


def load_prompt(prompt_file: str) -> dict:
    with open(prompt_file) as f:
        return json.load(f)


def build_system_prompt(prompt: dict) -> str:
    parts = [prompt["system_role"], ""]
    parts += [f"- {inst}" for inst in prompt["instructions"]]
    parts += ["", "Examples:"]
    for ex in prompt["examples"]:
        parts += [f"\nInput:\n{ex['input']}", f"Output: {json.dumps(ex['output'])}"]
    return "\n".join(parts)


def build_user_message(findings: str, indication: str, prompt: dict) -> str:
    parts = []
    if indication:
        parts.append(f"Indication: {indication}")
    parts.append(f"Findings:\n{findings}")
    input_text = "\n\n".join(parts)
    return prompt["final_instruction"].format(input_text=input_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap total samples before sharding (default: all)")
    parser.add_argument("--output", type=str, default="medgemma_summaries.jsonl")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="0-based shard index (use with --num-shards)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards (use with --shard-index)")
    args = parser.parse_args()

    if (args.shard_index is None) != (args.num_shards is None):
        parser.error("--shard-index and --num-shards must be provided together")
    if args.shard_index is not None and not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard-index must be in [0, num_shards-1], got {args.shard_index} with --num-shards {args.num_shards}")

    from vllm import LLM, SamplingParams

    prompt = load_prompt(PROMPT_FILE)
    system_prompt = build_system_prompt(prompt)

    print(f"Loading dataset split='{args.split}' ...")
    ds = load_rexgradient_dataset(split=args.split)
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    if args.shard_index is not None:
        n_total = len(ds)
        chunk_size = (n_total + args.num_shards - 1) // args.num_shards
        start = args.shard_index * chunk_size
        end = min(start + chunk_size, n_total)
        ds = ds.select(range(start, end))
        print(f"  Shard {args.shard_index}/{args.num_shards}: rows {start}–{end - 1} of {n_total}")

    print(f"  Processing {len(ds)} samples")

    findings_col = next(c for c in ds.column_names if "finding" in c.lower())
    indication_col = next((c for c in ds.column_names if "indication" in c.lower()), None)
    impression_col = next((c for c in ds.column_names if "impression" in c.lower()), None)

    messages_batch = []
    for i in range(len(ds)):
        findings = ds[i][findings_col]
        indication = ds[i][indication_col] if indication_col else None
        user_msg = build_user_message(findings, indication, prompt)
        messages_batch.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ])

    print(f"Loading model: {args.model_id}")
    llm = LLM(
        model=args.model_id,
        download_dir=args.cache_dir,
        max_model_len=8192,
        enable_prefix_caching=True,
        dtype="bfloat16",
        enforce_eager=True,          # disables CUDA graphs, avoids cuDNN init issues
        limit_mm_per_prompt={"image": 0},  # text-only — skip vision encoder init
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.95,
    )

    print("Generating classifications ...")
    outputs = llm.chat(messages_batch, sampling_params)

    output_path = args.output
    if args.shard_index is not None and args.output in ["medgemma_summaries.jsonl", "qwen3_summaries.jsonl"]:
        
        if "medgemma" in args.model_id.lower():
            stem = f"medgemma_summaries_{args.split}_shard{args.shard_index:04d}_of{args.num_shards:04d}.jsonl"
        elif "qwen" in args.model_id.lower():
            stem = f"qwen3_summaries_{args.split}_shard{args.shard_index:04d}_of{args.num_shards:04d}.jsonl"
        else:
            stem = f"summaries_{args.split}_shard{args.shard_index:04d}_of{args.num_shards:04d}.jsonl"
        
        output_path = os.path.join(os.path.dirname(__file__), "summaries", stem)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        for i, out in enumerate(outputs):
            raw_text = out.outputs[0].text.strip()

            # Attempt to parse the structured JSON output
            classification = None
            try:
                classification = json.loads(raw_text)
            except json.JSONDecodeError:
                # Model may have wrapped JSON in a code block
                if "```" in raw_text:
                    inner = raw_text.split("```")[1].lstrip("json").strip()
                    try:
                        classification = json.loads(inner)
                    except json.JSONDecodeError:
                        pass

            record = {
                "id": i,
                "findings": ds[i][findings_col],
                "classification": classification,
                "raw_output": raw_text if classification is None else None,
            }
            if indication_col:
                record["indication"] = ds[i][indication_col]
            if impression_col:
                record["reference_impression"] = ds[i][impression_col]

            f.write(json.dumps(record) + "\n")

    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    print(f"Saved {len(outputs)} records → {output_path}")


if __name__ == "__main__":
    main()
