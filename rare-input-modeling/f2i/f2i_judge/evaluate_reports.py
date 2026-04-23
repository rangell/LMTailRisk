"""
Evaluate radiology reports: compare ground truth from test_metadata.csv
against predicted reports using CRIMSON.

Usage examples:
  # Predicted column from CSV, first 50
  python evaluate_reports.py --pred-column Predicted --n 50

  # With detailed error counts
  python evaluate_reports.py --pred-column Predicted --details

  # Custom output path
  python evaluate_reports.py --pred-column Predicted --output results.json
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted radiology reports against ground truth."
    )
    parser.add_argument(
        "--input", default="data/rexgradient/test_metadata.csv",
        help="Path to ground truth CSV (default: data/rexgradient/test_metadata.csv)"
    )
    parser.add_argument(
        "--gt-column", default="Findings",
        help="Column name for ground truth reports (default: Findings)"
    )
    parser.add_argument(
        "--pred-column", required=True,
        help="Column name for predicted reports"
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Only evaluate the first N rows (default: all)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (.xlsx or .csv). Default: data/evaluation_results_<timestamp>.xlsx"
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Also output total/weighted error counts from CRIMSON"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Max concurrent CRIMSON API calls (default: 8)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of samples per forward pass for HuggingFace inference (default: 1)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.n:
        df = df.head(args.n)
    print(f"Loaded {len(df)} rows from {args.input}")

    gt_reports = df[args.gt_column].tolist()
    pred_reports = df[args.pred_column].tolist()
    pred_source = f"column '{args.pred_column}'"

    print(f"Ground truth: column '{args.gt_column}' | Predicted: {pred_source}")
    print(f"Evaluating {len(gt_reports)} pairs...")

    results = []
    for i in range(len(gt_reports)):
        results.append({
            "id": df.iloc[i].get("id", i),
            "ground_truth": gt_reports[i],
            "predicted": pred_reports[i],
        })

    from CRIMSON.generate_score import CRIMSONScore
    scorer = CRIMSONScore()

    print(f"\nComputing CRIMSON scores...")
    t0 = time.time()

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
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None

    def score_one(idx):
        row = results[idx]
        return idx, score_with_retry(row["ground_truth"], row["predicted"])

    n_success, n_fail = 0, 0

    if scorer.api in ["huggingface", "hf"]:
        batch_size = max(1, args.batch_size)
        print(f"  Using HuggingFace batched inference (batch_size={batch_size})")

        batch_starts = range(0, len(results), batch_size)
        for start in tqdm(batch_starts, desc="CRIMSON"):
            batch_indices = list(range(start, min(start + batch_size, len(results))))
            refs = [results[i]["ground_truth"] for i in batch_indices]
            preds = [results[i]["predicted"] for i in batch_indices]

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
                        results[idx]["crimson_json"] = None
                        n_fail += 1
                    else:
                        assign_crimson_result(idx, result)
                        n_success += 1
                continue

            for idx, result in zip(batch_indices, batch_result):
                assign_crimson_result(idx, result)
                n_success += 1
    else:
        print(f"  Using parallel API calls (max_workers={args.max_workers})")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(score_one, i) for i in range(len(results))]
            for future in tqdm(futures, desc="CRIMSON"):
                idx, result = future.result()
                if result is None:
                    results[idx]["crimson_score"] = None
                    results[idx]["crimson_json"] = None
                    n_fail += 1
                    continue
                assign_crimson_result(idx, result)
                n_success += 1

    print(f"  CRIMSON done in {time.time() - t0:.1f}s ({n_success} succeeded, {n_fail} failed)")
    if n_fail > 0:
        print(f"  WARNING: {n_fail} rows failed — average will exclude them!")

    print("\n" + "=" * 60)
    print("AVERAGE SCORES")
    print("=" * 60)

    avg_row = {}

    scores = [r["crimson_score"] for r in results if r.get("crimson_score") is not None]
    if scores:
        avg_val = sum(scores) / len(scores)
        avg_row["crimson_score"] = round(avg_val, 4)
        print(f"  crimson_score: {avg_val:.4f}")

    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"results/evaluation_results_{int(time.time())}.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "average": avg_row,
        "results": results,
    }

    def _json_default(obj):
        try:
            return obj.item()  # converts numpy scalars to native Python types
        except AttributeError:
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default) 

    print(f"\nResults saved to: {out_path}")
    print(f"Total rows: {len(results)}")


if __name__ == "__main__":
    main()