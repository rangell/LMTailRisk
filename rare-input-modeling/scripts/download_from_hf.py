import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset to disk")
    parser.add_argument("repo_id", help="HuggingFace repo ID, e.g. username/dataset-name")
    parser.add_argument("output_path", help="Local directory to save the dataset")
    parser.add_argument("--split", default=None, help="Specific split to download (default: all)")
    parser.add_argument("--token", default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    print(f"Downloading {args.repo_id} → {args.output_path} ...")
    ds = load_dataset(args.repo_id, split=args.split, token=args.token)
    print(ds)

    print(f"Saving to {args.output_path} ...")
    ds.save_to_disk(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
