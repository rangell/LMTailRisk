"""
Repack a folder of Arrow IPC stream shards into a clean HuggingFace dataset.

When a dataset is created via .filter() and saved, HuggingFace may embed
indices-mapping metadata in the Arrow files. Uploading those to the Hub causes:

    ArrowInvalidError: ... trying to cast schema to {'indices': Value('uint64')}

This script loads the shards directly via pyarrow (bypassing the cached filter
logic), constructs a fresh Dataset, and saves it to disk ready for push_to_hub.
"""

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset
from tqdm import tqdm


def load_shards(data_dir: str) -> pa.Table:
    shard_files = sorted(glob.glob(os.path.join(data_dir, "*.arrow")))
    if not shard_files:
        raise FileNotFoundError(f"No .arrow files found in {data_dir}")
    print(f"Loading {len(shard_files)} shards from {data_dir} ...")
    tables = []
    for path in tqdm(shard_files, desc="Loading shards", unit="shard"):
        with pa.memory_map(path, "r") as source:
            tables.append(ipc.open_stream(source).read_all())
    table = pa.concat_tables(tables)
    print(f"  Total rows: {len(table):,}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Folder containing the broken .arrow shards")
    parser.add_argument("output_dir", help="Where to write the repacked dataset")
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        default=None,
        help="HuggingFace repo id to push to after repacking (e.g. myorg/myrepo)",
    )
    args = parser.parse_args()

    table = load_shards(args.data_dir)

    # Constructing Dataset from a raw pa.Table drops the broken HF metadata
    # and rebuilds clean features from the actual schema.
    print("Building clean Dataset ...")
    ds = Dataset(table)

    print(f"Saving to {args.output_dir} ...")
    ds.save_to_disk(args.output_dir)
    print("Done.")

    if args.push_to_hub:
        print(f"Pushing to Hub: {args.push_to_hub} ...")
        ds.push_to_hub(args.push_to_hub)
        print("Push complete.")


if __name__ == "__main__":
    main()
