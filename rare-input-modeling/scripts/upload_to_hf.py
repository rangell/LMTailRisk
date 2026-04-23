import argparse
import glob
import os
from datasets import load_from_disk, Dataset, concatenate_datasets


def main():
    parser = argparse.ArgumentParser(description="Upload a local HF dataset to the Hub")
    parser.add_argument("dataset_path", help="Path to the local dataset directory")
    parser.add_argument("repo_id", help="HuggingFace repo ID, e.g. username/dataset-name")
    parser.add_argument("--private", action="store_true", default=False, help="Make the dataset private")
    parser.add_argument("--token", default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path} ...")
    if os.path.exists(os.path.join(args.dataset_path, "dataset_info.json")):
        ds = load_from_disk(args.dataset_path)
    else:
        arrow_files = sorted(glob.glob(os.path.join(args.dataset_path, "*.arrow")))
        if not arrow_files:
            raise FileNotFoundError(f"No Arrow files found in {args.dataset_path}")
        print(f"Found {len(arrow_files)} Arrow files, loading directly ...")
        ds = concatenate_datasets([Dataset.from_file(f) for f in arrow_files])
    print(ds)

    print(f"Uploading to {args.repo_id} (private={args.private}) ...")
    ds.push_to_hub(args.repo_id, private=args.private, token=args.token)
    print("Done.")


if __name__ == "__main__":
    main()
