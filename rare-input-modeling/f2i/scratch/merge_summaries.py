"""
Check that all 20 shards are present for each split, then merge into one
JSONL per split.

Usage:
    python merge_summaries.py                        # dry run (check only)
    python merge_summaries.py --merge                # check + merge complete splits
    python merge_summaries.py --merge --force        # merge even if shards are missing
"""

import argparse
import json
import os

SUMMARIES_DIR = os.path.join(os.path.dirname(__file__), "summaries")
SPLITS = ["train", "validation", "test"]
NUM_SHARDS = 20


def shard_path(split: str, shard_index: int) -> str:
    fname = f"medgemma_summaries_{split}_shard{shard_index:04d}_of{NUM_SHARDS:04d}.jsonl"
    return os.path.join(SUMMARIES_DIR, fname)


def check_shards(split: str) -> list[int]:
    """Return list of missing shard indices."""
    return [i for i in range(NUM_SHARDS) if not os.path.exists(shard_path(split, i))]


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def merge_split(split: str, output_path: str) -> int:
    total = 0
    with open(output_path, "w") as out:
        for i in range(NUM_SHARDS):
            path = shard_path(split, i)
            with open(path) as f:
                for line in f:
                    out.write(line)
                    total += 1
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true",
                        help="Merge complete splits (default: check only)")
    parser.add_argument("--force", action="store_true",
                        help="Merge even if some shards are missing")
    args = parser.parse_args()

    all_ok = True
    for split in SPLITS:
        missing = check_shards(split)
        present = NUM_SHARDS - len(missing)
        if missing:
            all_ok = False
            print(f"[{split:>10}]  INCOMPLETE  {present}/{NUM_SHARDS} shards present — missing: {missing}")
        else:
            shard_sizes = [count_lines(shard_path(split, i)) for i in range(NUM_SHARDS)]
            total = sum(shard_sizes)
            print(f"[{split:>10}]  OK          {NUM_SHARDS}/{NUM_SHARDS} shards present, {total:,} rows total")

    print()

    if not args.merge:
        print("Run with --merge to merge complete splits.")
        return

    for split in SPLITS:
        missing = check_shards(split)
        if missing and not args.force:
            print(f"[{split:>10}]  SKIP (missing shards {missing}). Use --force to merge anyway.")
            continue

        output_path = os.path.join(SUMMARIES_DIR, f"medgemma_summaries_{split}.jsonl")
        print(f"[{split:>10}]  Merging → {output_path} ...", end=" ", flush=True)
        n = merge_split(split, output_path)
        print(f"{n:,} rows written.")


if __name__ == "__main__":
    main()
