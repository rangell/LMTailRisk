import numpy as np
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy.stats import beta as beta_dist


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(results_dir: str = "./results", split: str = "train") -> list[dict]:
    eval_paths = list(Path(results_dir).rglob(f"crimson_{split}*.jsonl"))
    data = []
    for eval_path in eval_paths:
        assert eval_path.is_file(), f"Expected file but found {eval_path}"
        print(eval_path)
        with open(eval_path) as f:
            data.extend(json.load(f)["results"])
    print(len(data), "samples loaded from", len(eval_paths), "files")
    return data


def load_data_by_split(results_dir: str = "./results") -> dict[str, list[dict]]:
    """Load results separately for train, val, and test splits."""
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for path in sorted(Path(results_dir).rglob("crimson_*.jsonl")):
        assert path.is_file()
        with open(path) as f:
            results = json.load(f)["results"]
        if path.name.startswith("crimson_train"):
            splits["train"].extend(results)
        elif path.name.startswith("crimson_val"):
            splits["val"].extend(results)
        else:
            splits["test"].extend(results)
    for name, rows in splits.items():
        print(f"  {name}: {len(rows)} samples")
    return splits


def extract_metrics(data: list[dict]) -> dict:
    """Return per-sample arrays extracted from CRIMSON results."""
    n_missed_findings = []
    n_urgent_missed_findings = []
    n_samples_with_urgent = []

    for sample in data:
        try:
            missed = sample["raw_evaluation"]["errors"]["missing_findings"]
        except KeyError:
            missed = []

        # missed entries can be strings (ids) or dicts with an "id" key
        missed_ids = {m if isinstance(m, str) else m.get("id", "") for m in missed}

        n_missed_findings.append(len(missed_ids))

        n_urgent = 0
        for ref in sample.get("raw_evaluation", {}).get("reference_findings", []):
            if ref["id"] in missed_ids and ref.get("clinical_significance") == "urgent":
                n_urgent += 1
        n_urgent_missed_findings.append(n_urgent)
        n_samples_with_urgent.append(int(n_urgent > 0))

    # import pdb; pdb.set_trace()
    return {
        "n_missed_findings": np.array(n_missed_findings),
        "n_urgent_missed_findings": np.array(n_urgent_missed_findings),
        "n_samples_with_urgent": np.array(n_samples_with_urgent),
    }


# ---------------------------------------------------------------------------
# Urgent missed findings export
# ---------------------------------------------------------------------------

def save_urgent_missed_findings(data: list[dict], out_path: str = "./scratch/urgent_missed_findings.csv") -> pd.DataFrame:
    """Extract all urgent missed findings and save to CSV.

    Each row is one urgent missed finding and contains:
      - all top-level scalar fields from the sample (input, output, etc.)
      - all fields from the reference_findings entry (id, finding, clinical_significance, ...)
      - all fields from the missed-finding entry (explanation, etc.) prefixed with "missed_"
      - all scalar fields from raw_evaluation (e.g. overall explanation) prefixed with "crimson_"
    """
    rows = []
    for i, sample in enumerate(data):
        raw = sample.get("raw_evaluation", {})
        missed = raw.get("errors", {}).get("missing_findings", [])

        # Build a map from id -> missed entry (which may contain an explanation)
        missed_map: dict[str, dict] = {}
        for m in missed:
            if isinstance(m, str):
                missed_map[m] = {}
            else:
                missed_map[m.get("id", "")] = m

        # Top-level scalar/string fields (input report, model output, metadata, etc.)
        sample_fields = {
            k: v for k, v in sample.items()
            if k != "raw_evaluation" and not isinstance(v, (dict, list))
        }

        # Scalar fields from raw_evaluation (e.g. overall explanation or score)
        crimson_fields = {
            f"crimson_{k}": v for k, v in raw.items()
            if k not in ("reference_findings", "errors") and not isinstance(v, (dict, list))
        }

        for ref in raw.get("reference_findings", []):
            if ref["id"] in missed_map and ref.get("clinical_significance") == "urgent":
                row = {"sample_index": i}
                row.update(sample_fields)
                row.update(crimson_fields)
                # Reference finding fields
                row.update(ref)
                # Missed-finding entry fields (e.g. explanation), prefixed to avoid collisions
                for k, v in missed_map[ref["id"]].items():
                    row[f"missed_{k}"] = v
                rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} urgent missed findings to {out}")
    return df


# ---------------------------------------------------------------------------
# Clopper-Pearson CI analysis
# ---------------------------------------------------------------------------

def clopper_pearson_ci(k: int, n: int, ci: float = 0.95) -> tuple[float, float, float]:
    """Exact Clopper-Pearson CI for a proportion.

    k: number of successes
    n: number of trials
    Returns (proportion, lower, upper).
    """
    alpha = (1 - ci) / 2
    p_hat = k / n
    lower = float(beta_dist.ppf(alpha, k, n - k + 1)) if k > 0 else 0.0
    upper = float(beta_dist.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0
    return float(p_hat), lower, upper


def sample_size_analysis(
    values: np.ndarray,
    metric_name: str = "prop_samples_with_urgent",
    n_steps: int = 30,
    ci: float = 0.95,
    convergence_threshold=None,
    seed: int = 42,
    out_dir: str = "./results",
    min_n: int = 100,
    max_n: int = None,
) -> pd.DataFrame:
    """Sweep sample sizes from min_n to len(values), compute Clopper-Pearson CIs.

    values must be a binary (0/1) array. At each n, a subsample of size n is
    drawn without replacement and the exact binomial CI is computed for the
    observed proportion of successes.

    Returns a DataFrame with columns: n, mean, lower, upper, ci_width.
    Saves a plot to out_dir.
    """
    rng = np.random.default_rng(seed)
    N = min(len(values), max_n) if max_n is not None else len(values)

    # Log-spaced n values so small-n behaviour is well-resolved
    ns = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(N), n_steps)).astype(int))
    ns = ns[ns <= N]

    rows = []
    for n in ns:
        subset = rng.choice(values, size=n, replace=False)
        k = int(subset.sum())
        p_hat, lower, upper = clopper_pearson_ci(k, n, ci=ci)
        rows.append({"n": int(n), "mean": p_hat, "lower": lower, "upper": upper,
                     "ci_width": upper - lower})
        print(f"  n={n:5d}  p={p_hat:.4f}  {int(ci*100)}% CI=[{lower:.4f}, {upper:.4f}]  width={upper-lower:.4f}")

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Find convergence point
    # ------------------------------------------------------------------
    full_p = float(values.mean())
    if convergence_threshold is None:
        # default: CI width <= 10% of the full-data proportion (or 0.05 if p≈0)
        convergence_threshold = max(0.05, 0.10 * abs(full_p))

    converged = df[df["ci_width"] <= convergence_threshold]
    n_converge = int(converged["n"].iloc[0]) if not converged.empty else None
    print(f"\nFull-data proportion ({N} samples): {full_p:.4f}")
    if n_converge:
        print(f"CI width ≤ {convergence_threshold:.4f} first achieved at n={n_converge}")
    else:
        print(f"CI width never reached threshold {convergence_threshold:.4f} within {N} samples")

    # ------------------------------------------------------------------
    # Plot: proportion + CI ribbon (top) | CI width (bottom)
    # ------------------------------------------------------------------
    fig, (ax, ax_width) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax.fill_between(
        df["n"], df["lower"], df["upper"],
        alpha=0.25, color="C0",
        label=f"Clopper-Pearson {int(ci * 100)}% CI",
    )
    ax.plot(df["n"], df["mean"], color="C0", lw=2, label="Observed proportion")
    ax.axhline(
        full_p, color="black", lw=1.5, ls="--",
        label=f"Proportion on all {N} samples ({full_p * 100:.1f}%)",
    )
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        f"Clopper-Pearson {int(ci * 100)}% CI  —  {metric_name}",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    ax_width.plot(df["n"], df["ci_width"], color="C1", lw=2)
    if n_converge is not None:
        ax_width.axvline(n_converge, color="gray", lw=1.2, ls=":", label=f"Converges at n={n_converge}")
        ax_width.axhline(convergence_threshold, color="gray", lw=1.2, ls="--", label=f"Threshold ({convergence_threshold:.4f})")
        ax_width.legend(fontsize=9)
    ax_width.set_xscale("log")
    ax_width.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_width.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.2f}%"))
    ax_width.set_xticks(ns)
    ax_width.set_xlabel("Number of samples (n)", fontsize=12)
    ax_width.set_ylabel(f"{int(ci * 100)}% CI width", fontsize=12)
    ax_width.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    out_path = Path(out_dir) / f"sample_size_analysis_{metric_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Split comparison plot
# ---------------------------------------------------------------------------

def plot_split_comparison(
    splits: dict[str, np.ndarray],
    metric_name: str = "prop_samples_with_urgent",
    n_steps: int = 30,
    ci: float = 0.95,
    seed: int = 42,
    out_dir: str = "./results",
    min_n: int = 100,
    max_n: int = None,
) -> dict[str, pd.DataFrame]:
    """Compute Clopper-Pearson CIs for each split and overlay them on one plot.

    splits: mapping of label -> binary (0/1) np.ndarray
    Returns a dict of label -> DataFrame with columns n, mean, lower, upper, ci_width.
    """
    colors = {"train": "C0", "test": "C1"}
    labels = {"train": "Train", "test": "Test"}

    fig, (ax, ax_width) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    dfs: dict[str, pd.DataFrame] = {}

    for split_name, values in splits.items():
        rng = np.random.default_rng(seed)
        N = min(len(values), max_n) if max_n is not None else len(values)
        ns = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(N), n_steps)).astype(int))
        ns = ns[ns <= N]

        rows = []
        for n in ns:
            subset = rng.choice(values, size=n, replace=False)
            k = int(subset.sum())
            p_hat, lower, upper = clopper_pearson_ci(k, n, ci=ci)
            rows.append({"n": int(n), "mean": p_hat, "lower": lower, "upper": upper,
                         "ci_width": upper - lower})

        df = pd.DataFrame(rows)
        dfs[split_name] = df
        full_p = float(values.mean())
        color = colors.get(split_name, None)
        label = labels.get(split_name, split_name)

        ax.fill_between(df["n"], df["lower"], df["upper"], alpha=0.20, color=color)
        ax.plot(df["n"], df["mean"], color=color, lw=2, label=f"{label} (n={N}, p={full_p*100:.1f}%)")
        ax.axhline(full_p, color=color, lw=1.2, ls="--")

        ax_width.plot(df["n"], df["ci_width"], color=color, lw=2, label=label)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        f"Clopper-Pearson {int(ci * 100)}% CI by split  —  {metric_name}",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    x_ticks = [1000, 5000, 10000, 50000, 100000]
    ax_width.set_xscale("log")
    ax_width.set_xticks(x_ticks)
    ax_width.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_width.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.2f}%"))
    ax_width.set_xlabel("Number of samples (n)", fontsize=12)
    ax_width.set_ylabel(f"{int(ci * 100)}% CI width", fontsize=12)
    ax_width.legend(fontsize=10)
    ax_width.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    out_path = Path(out_dir) / f"split_comparison_{metric_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)

    return dfs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- load splits separately ---
    print("Loading data by split...")
    splits_data = load_data_by_split("./results")

    split_metrics = {name: extract_metrics(rows) for name, rows in splits_data.items()}

    for name, metrics in split_metrics.items():
        n_urgent_binary = metrics["n_samples_with_urgent"]
        n = len(n_urgent_binary)
        print(f"\n[{name}] n={n}  samples with urgent missed: "
              f"{n_urgent_binary.sum()} ({n_urgent_binary.mean()*100:.1f}%)")

    # --- split comparison plot ---
    print("\n" + "=" * 60)
    print("Split comparison: train vs test")
    print("=" * 60)
    split_binaries = {name: m["n_samples_with_urgent"] for name, m in split_metrics.items()
                      if name in ("train", "test")}
    dfs = plot_split_comparison(
        splits=split_binaries,
        metric_name="prop_samples_with_urgent",
        n_steps=50,
        ci=0.99,
        out_dir="./scratch",
        min_n=1000,
        max_n=100_000,
    )
    for name, df in dfs.items():
        csv_path = Path(f"./scratch/split_comparison_{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Table saved to {csv_path}")

    # --- export urgent missed findings (test only) ---
    save_urgent_missed_findings(splits_data["test"], out_path="./scratch/urgent_missed_findings.csv")


if __name__ == "__main__":
    main()
