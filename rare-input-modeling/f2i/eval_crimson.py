"""
Evaluate MedGemma F2I predictions using CRIMSON.

Reads the JSONL produced by generate_responses.py, where each line contains:
  - predicted_impression  (model output)
  - reference_impression  (ground truth)

Results are checkpointed every SAVE_EVERY batches so progress is not lost
on long runs.

Usage:
  python eval_crimson.py --input results/medgemma_f2i.jsonl
  python eval_crimson.py --input results/medgemma_f2i.jsonl --details --n 200
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

sys.path.append('f2i_judge')

from CRIMSON.generate_score import CRIMSONScore

SAVE_EVERY = 5  # checkpoint every N batches


def load_jsonl(path: str, n: int | None) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if n is not None and len(records) >= n:
                break
    return records


def _json_default(obj):
    try:
        return obj.item()
    except AttributeError:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_checkpoint(out_path: Path, results: list[dict]) -> None:
    scores = [r["crimson_score"] for r in results if r.get("crimson_score") is not None]
    avg_row = {}
    if scores:
        avg_row["crimson_score"] = round(sum(scores) / len(scores), 4)
    with open(out_path, "w") as f:
        json.dump({"average": avg_row, "results": results}, f, indent=2, default=_json_default)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generate_responses.py JSONL output with CRIMSON."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to JSONL file produced by generate_responses.py",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Only evaluate the first N rows (default: all)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: results/evaluation_results_<timestamp>.json)",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Also output per-category error counts from CRIMSON",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Max concurrent CRIMSON API calls (default: 8)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for HuggingFace CRIMSON inference (default: 1)",
    )
    parser.add_argument(
        "--job-index", type=int, default=None,
        help="0-based index of this job (use with --num-jobs)",
    )
    parser.add_argument(
        "--num-jobs", type=int, default=None,
        help="Total number of parallel jobs (use with --job-index)",
    )
    args = parser.parse_args()

    if (args.job_index is None) != (args.num_jobs is None):
        parser.error("--job-index and --num-jobs must be provided together")
    if args.job_index is not None and not (0 <= args.job_index < args.num_jobs):
        parser.error(f"--job-index must be in [0, num_jobs-1], got {args.job_index} with --num-jobs {args.num_jobs}")

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    print(f"Loading predictions from {args.input} ...")
    raw = load_jsonl(args.input, args.n)

    missing_ref = sum(1 for r in raw if not r.get("reference_impression"))
    if missing_ref == len(raw):
        print("ERROR: no 'reference_impression' field found — cannot evaluate.", file=sys.stderr)
        sys.exit(1)
    if missing_ref:
        print(f"  WARNING: {missing_ref}/{len(raw)} rows have no reference_impression and will be skipped.")

    results = []
    for r in raw:
        ref = r.get("reference_impression", "")
        pred = r.get("predicted_impression", "")
        if not ref or not pred:
            continue
        results.append({
            "id": r.get("id", len(results)),
            "ground_truth": ref,
            "predicted": pred,
            "findings": r.get("findings"),
        })

    # ------------------------------------------------------------------
    # Slice for this job
    # ------------------------------------------------------------------
    if args.job_index is not None:
        n_total = len(results)
        chunk_size = (n_total + args.num_jobs - 1) // args.num_jobs
        start = args.job_index * chunk_size
        end = min(start + chunk_size, n_total)
        print(f"  Job {args.job_index}/{args.num_jobs}: rows {start}–{end - 1} of {n_total}")
        results = results[start:end]

    print(f"  Evaluating {len(results)} pairs ...")

    if args.output:
        out_path = Path(args.output)
    elif args.job_index is not None:
        out_path = Path(f"results/evaluation_results_job{args.job_index:04d}_of{args.num_jobs:04d}.json")
    else:
        out_path = Path(f"results/evaluation_results_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRIMSON scoring
    # ------------------------------------------------------------------
    scorer = CRIMSONScore()

    def assign_crimson_result(idx, result):
        results[idx]["crimson_score"] = result["crimson_score"]
        results[idx]["raw_evaluation"] = result["raw_evaluation"]
        if args.details:
            ec = result.get("error_counts", {})
            results[idx]["crimson_total_errors"] = (
                ec.get("false_findings", 0)
                + ec.get("missing_findings", 0)
                + ec.get("attribute_errors", 0)
            )
            wc = result.get("weighted_error_counts", {})
            results[idx]["crimson_weighted_errors"] = (
                wc.get("false_findings", 0)
                + wc.get("missing_findings", 0)
                + wc.get("attribute_errors", 0)
            )
            results[idx]["crimson_false_findings"] = ec.get("false_findings", 0)
            results[idx]["crimson_missing_findings"] = ec.get("missing_findings", 0)
            results[idx]["crimson_attribute_errors"] = ec.get("attribute_errors", 0)

    def score_with_retry(gt, pred):
        for attempt in range(3):
            try:
                return scorer.evaluate(gt, pred)
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    def score_one(idx):
        row = results[idx]
        return idx, score_with_retry(row["ground_truth"], row["predicted"])

    print("\nComputing CRIMSON scores ...")
    t0 = time.time()
    n_success, n_fail = 0, 0

    if scorer.api in ["huggingface", "hf"]:
        batch_size = max(1, args.batch_size)
        print(f"  HuggingFace batched inference (batch_size={batch_size})")
        batch_starts = list(range(0, len(results), batch_size))
        for batch_num, start in enumerate(tqdm(batch_starts, desc="CRIMSON")):
            batch_indices = list(range(start, min(start + batch_size, len(results))))
            refs  = [results[i]["ground_truth"] for i in batch_indices]
            preds = [results[i]["predicted"]     for i in batch_indices]

            batch_result = None
            for attempt in range(3):
                try:
                    batch_result = scorer.evaluate_batch(refs, preds, batch_size=batch_size)
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)

            if batch_result is None:
                for idx in batch_indices:
                    result = score_with_retry(results[idx]["ground_truth"], results[idx]["predicted"])
                    if result is None:
                        results[idx]["crimson_score"] = None
                        n_fail += 1
                    else:
                        assign_crimson_result(idx, result)
                        n_success += 1
            else:
                for idx, result in zip(batch_indices, batch_result):
                    if result is None:
                        results[idx]["crimson_score"] = None
                        n_fail += 1
                    else:
                        assign_crimson_result(idx, result)
                        n_success += 1

            if (batch_num + 1) % SAVE_EVERY == 0:
                save_checkpoint(out_path, results)
                tqdm.write(f"  [checkpoint] saved after batch {batch_num + 1} → {out_path}")
    else:
        print(f"  Parallel API calls (max_workers={args.max_workers})")
        # For the API path, group completed futures into batches of batch_size
        # for checkpoint cadence purposes.
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(score_one, i) for i in range(len(results))]
            for completed, future in enumerate(tqdm(futures, desc="CRIMSON")):
                idx, result = future.result()
                if result is None:
                    results[idx]["crimson_score"] = None
                    n_fail += 1
                else:
                    assign_crimson_result(idx, result)
                    n_success += 1

                # treat each future as 1 "unit"; checkpoint every SAVE_EVERY units
                if (completed + 1) % (SAVE_EVERY * max(1, args.batch_size)) == 0:
                    save_checkpoint(out_path, results)
                    tqdm.write(f"  [checkpoint] saved after {completed + 1} samples → {out_path}")

    print(f"  Done in {time.time() - t0:.1f}s ({n_success} succeeded, {n_fail} failed)")
    if n_fail:
        print(f"  WARNING: {n_fail} rows failed — average excludes them.")

    # ------------------------------------------------------------------
    # Final save + summary
    # ------------------------------------------------------------------
    save_checkpoint(out_path, results)

    scores = [r["crimson_score"] for r in results if r.get("crimson_score") is not None]
    print("\n" + "=" * 60)
    print("AVERAGE SCORES")
    print("=" * 60)
    if scores:
        avg_val = sum(scores) / len(scores)
        print(f"  crimson_score: {avg_val:.4f}  (n={len(scores)})")
    print("=" * 60)

    print(f"\nResults saved to: {out_path}  ({len(results)} rows)")


if __name__ == "__main__":
    main()