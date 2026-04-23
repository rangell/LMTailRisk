"""Calculate CRIMSON and RadEval scores for RadPref preference data.

Two-phase pipeline:
  Phase 1 – CRIMSON  (I/O-bound API calls → ThreadPoolExecutor)
  Phase 2 – RadEval  (GPU-bound local models → sequential per-sample)
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from CRIMSON.generate_score import CRIMSONScore

try:
    from RadEval.RadEval import RadEval
    RADEVAL_AVAILABLE = True
except Exception:
    RADEVAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def flatten_radeval_scores(scores):
    """Flatten RadEval output dict into a single-level {metric: value} dict."""
    flat = {}
    for key, value in scores.items():
        if key.startswith('rouge') and key != 'rougeL':
            continue
        if key.startswith('chexbert') and key != 'chexbert-5_micro avg_f1-score':
            continue
        if key.startswith('radgraph') and key not in ['radgraph_complete']:
            continue
        if isinstance(value, dict):
            if 'bleu' in key.lower():
                flat[key] = value.get('bleu-4', list(value.values())[0])
            elif 'rouge' in key.lower():
                flat['rougeL'] = value.get('rougeL', value.get('rouge-l', list(value.values())[0]))
            elif 'chexbert' in key.lower():
                flat['chexbert'] = value.get('chexbert-5_micro avg_f1-score', list(value.values())[0])
            else:
                flat[key] = list(value.values())[0]
        else:
            flat[key] = value
    return flat


def evaluate_with_retry(scorer, gt, candidate, patient_context, max_retries=3,
                        include_guidelines=True):
    for attempt in range(max_retries):
        try:
            return scorer.evaluate(gt, candidate, patient_context=patient_context,
                                   include_guidelines=include_guidelines)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


# ---------------------------------------------------------------------------
# Phase 1 – CRIMSON 
# ---------------------------------------------------------------------------
def _crimson_task(scorer, gt, candidate, patient_context, include_guidelines=True):
    """Single CRIMSON evaluation (designed to run in a thread)."""
    return evaluate_with_retry(scorer, gt, candidate, patient_context,
                               include_guidelines=include_guidelines)


def _findings_to_paragraph(findings):
    """Join a list of finding strings into a single paragraph."""
    sentences = []
    for f in findings:
        f = f.strip()
        if not f:
            continue
        if not f.endswith("."):
            f += "."
        sentences.append(f)
    return " ".join(sentences)


def run_crimson_parallel(data, scorer, workers=16, findings_mode=False,
                         include_guidelines=True):
    """Score all C1/C2 pairs with CRIMSON using a thread pool.

    Returns dict[sample_index] → {crimson_C1, crimson_C2,
                                   crimson_C1_full, crimson_C2_full}.
    """
    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, sample in enumerate(data):
            ctx = {
                "Age": sample["PatientAge"],
                "Sex": sample["PatientSex"],
                "Indication": sample["Indication"],
            }
            if findings_mode:
                gt = _findings_to_paragraph(sample["ground_truth_findings"])
                c1 = _findings_to_paragraph(sample["C1_findings"])
                c2 = _findings_to_paragraph(sample["C2_findings"])
            else:
                gt = sample["ground_truth_report"]
                c1 = sample["C1_report"]
                c2 = sample["C2_report"]
            futures[pool.submit(_crimson_task, scorer, gt, c1, ctx, include_guidelines)] = (idx, "C1")
            futures[pool.submit(_crimson_task, scorer, gt, c2, ctx, include_guidelines)] = (idx, "C2")

        pbar = tqdm(total=len(futures), desc="CRIMSON scoring")
        for future in as_completed(futures):
            idx, label = futures[future]
            res = future.result()
            entry = results.setdefault(idx, {})
            entry[f"crimson_{label}"] = res["crimson_score"]
            entry[f"crimson_{label}_full"] = res
            pbar.update(1)
        pbar.close()

    return results


# ---------------------------------------------------------------------------
# Phase 2 – RadEval 
# ---------------------------------------------------------------------------
def run_radeval(data, scorer, findings_mode=False):
    """Score all C1/C2 pairs with RadEval (one sample at a time).

    Returns (dict[sample_index] → {metric_C1: val, metric_C2: val, ...},
             list of metric names).
    """
    results = {}
    metric_names = []

    for idx, sample in enumerate(tqdm(data, desc="RadEval scoring")):
        if findings_mode:
            gt = _findings_to_paragraph(sample["ground_truth_findings"])
            c1 = _findings_to_paragraph(sample["C1_findings"])
            c2 = _findings_to_paragraph(sample["C2_findings"])
        else:
            gt = sample["ground_truth_report"]
            c1 = sample["C1_report"]
            c2 = sample["C2_report"]
        c1_raw = scorer(refs=[gt], hyps=[c1])
        c2_raw = scorer(refs=[gt], hyps=[c2])

        c1_flat = flatten_radeval_scores(c1_raw)
        c2_flat = flatten_radeval_scores(c2_raw)

        if idx == 0:
            metric_names = list(c1_flat.keys())

        entry = {}
        for key in c1_flat:
            entry[f"{key}_C1"] = c1_flat[key]
            entry[f"{key}_C2"] = c2_flat[key]
        results[idx] = entry

    return results, metric_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--skip-crimson", action="store_true", help="Skip CRIMSON scoring")
    parser.add_argument("--skip-radeval", action="store_true", help="Skip RadEval scoring")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for model weights")
    parser.add_argument("--workers", type=int, default=16, help="Threads for parallel CRIMSON API calls")
    parser.add_argument("--findings-mode", action="store_true",
                        help="Use findings lists instead of full reports")
    parser.add_argument("--no-guidelines", action="store_true",
                        help="Disable guidelines in the CRIMSON GPT prompt")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # ------------------------------------------------------------------
    # Phase 1 – CRIMSON
    # ------------------------------------------------------------------
    crimson_results = {}
    if not args.skip_crimson:
        print(f"\n=== Phase 1: CRIMSON ({args.workers} threads) ===")
        crimson_scorer = CRIMSONScore(api="openai", model_name="gpt-5.2")
        crimson_results = run_crimson_parallel(data, crimson_scorer, workers=args.workers,
                                                findings_mode=args.findings_mode,
                                                include_guidelines=not args.no_guidelines)

    # ------------------------------------------------------------------
    # Phase 2 – RadEval
    # ------------------------------------------------------------------
    radeval_results = {}
    radeval_metric_names = []
    if not args.skip_radeval:
        if not RADEVAL_AVAILABLE:
            print("\n⚠ RadEval is not installed. Skipping RadEval scoring.")
            print("  Install it with: pip install radeval")
        else:
            print("\n=== Phase 2: RadEval ===")
            radeval_kwargs = dict(
                do_radgraph=True,
                do_bleu=True,
                do_bertscore=True,
                do_green=True,
                do_chexbert=True,
                do_ratescore=True,
                do_rouge=True,
            )
            if args.cache_dir:
                radeval_kwargs['cache_dir'] = args.cache_dir
            radeval_scorer = RadEval(**radeval_kwargs)
            radeval_results, radeval_metric_names = run_radeval(data, radeval_scorer,
                                                                findings_mode=args.findings_mode)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    results = []
    for idx, sample in enumerate(data):
        result = {**sample}
        result.update(crimson_results.get(idx, {}))
        result.update(radeval_results.get(idx, {}))
        results.append(result)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {args.output}")

    # Print averages
    print("\nAverages:")
    metric_keys = (["crimson"] if crimson_results else []) + radeval_metric_names
    for key in metric_keys:
        c1_vals = [r[f"{key}_C1"] for r in results if f"{key}_C1" in r]
        c2_vals = [r[f"{key}_C2"] for r in results if f"{key}_C2" in r]
        if c1_vals and c2_vals:
            print(f"  {key}: C1={sum(c1_vals)/len(c1_vals):.4f}, C2={sum(c2_vals)/len(c2_vals):.4f}")


if __name__ == "__main__":
    main()