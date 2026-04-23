import os
import json
from collections import defaultdict

import torch

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample

from inspect_ai.model import (
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    ChatMessageUser,
    ChatMessageSystem,
    GenerateConfig,
)
from inspect_ai.solver import TaskState, solver, Generate

from petri.scorers import alignment_judge, DIMENSIONS
from petri.stores import AuditStore
from petri.transcript.transcript import TranscriptEvent, AddMessage


import pyarrow as pa

from prompts import JUDGE_PROMPT
from vllm_utils import vllm_server, find_free_port

JUDGE_PORT = find_free_port()
os.environ["OPENAI_BASE_URL"] = f"http://localhost:{JUDGE_PORT}/v1"
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

ROLE_MAP = {
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
    "system": ChatMessageSystem,
}

TRAITS = [
    "unprompted_deception_toward_user",
    "unfaithful_thinking",
    "unprompted_sycophancy",
    "unprompted_encouragement_of_user_delusion",
]


def hf_to_message(turn: dict):
    role = turn["role"]
    content = turn["content"]

    if role == "assistant" and turn.get("thinking") and turn["thinking"] != "":
        return ChatMessageAssistant(
            content=[
                ContentReasoning(reasoning=turn["thinking"]),
                ContentText(text=content),
            ]
        )

    return ROLE_MAP[role](content=content)


def hf_to_events(conversation) -> list[TranscriptEvent]:
    import json
    import warnings

    if conversation is None:
        warnings.warn("generated_conversation is None; returning empty events")
        return []
    if isinstance(conversation, str):
        warnings.warn("generated_conversation is a string; parsing as JSON")
        conversation = json.loads(conversation)
    return [
        TranscriptEvent(
            view="target",
            edit=AddMessage(message=hf_to_message(turn)),
        )
        for turn in conversation
    ]


@solver
def load_conversation():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        audit_store = state.store_as(AuditStore)
        # conversation is stashed in metadata by the Sample
        audit_store.events = hf_to_events(state.metadata["generated_conversation"])
        return state

    return solve


def parse_shards(shard_args: list[str], total: int) -> list[int]:
    """Parse shard specs like '0', '1-5', '3' into a sorted list of indices."""
    indices = set()
    for s in shard_args:
        if "-" in s:
            lo, hi = s.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(s))
    return sorted(i for i in indices if 0 <= i < total)


def load_conversations_from_shards(data_dir, shards=None):
    import glob
    import pyarrow as pa

    all_shard_files = sorted(glob.glob(os.path.join(data_dir, "*.arrow")))
    if not all_shard_files:  # HuggingFace datasets store shards in a train/ subdir
        all_shard_files = sorted(glob.glob(os.path.join(data_dir, "train", "*.arrow")))
    if shards is not None:
        indices = parse_shards(shards, len(all_shard_files))
        shard_files = [all_shard_files[i] for i in indices]
    else:
        shard_files = all_shard_files
    rows = []
    for shard_file in shard_files:
        shard_name = os.path.basename(shard_file)
        for row in pa.ipc.open_stream(shard_file).read_all().to_pylist():
            row["_shard"] = shard_name
            rows.append(row)
    return rows


def save_results(log, dimensions: dict, output_dir: str, conversations: list):

    dim_names = list(dimensions.keys())

    # Build lookup from conversation_hash -> original row (drop internal _shard key)
    originals = {
        r["conversation_hash"]: {k: v for k, v in r.items() if k != "_shard"}
        for r in conversations
    }

    by_shard = defaultdict(list)
    for sample in log.samples or []:
        scores = sample.scores or {}
        score = next(iter(scores.values()), None)
        if score is None:
            continue
        h = sample.metadata.get("conversation_hash", "")
        row = dict(originals.get(h, {"conversation_hash": h}))
        row["summary"] = score.answer or ""
        row["justification"] = score.explanation or ""
        row["highlights"] = json.dumps(score.metadata.get("highlights", []))
        row["raw_response"] = score.metadata.get("response", "")
        for dim in dim_names:
            row[dim] = int((score.value or {}).get(dim, -1))
        by_shard[sample.metadata.get("_shard", "unknown.arrow")].append(row)

    os.makedirs(output_dir, exist_ok=True)
    print("status:", log.status)

    print("samples:", len(log.samples or []))
    for s in (log.samples or [])[:2]:
        print("  scores keys:", list((s.scores or {}).keys()))
        print("  score value:", next(iter((s.scores or {}).values()), None))

    if not by_shard:
        print("No results to save.")
        return

    for shard_name, rows in by_shard.items():
        table = pa.Table.from_pylist(rows)
        # Ensure score columns are int32
        for dim in dim_names:
            if dim in table.schema.names:
                table = table.set_column(
                    table.schema.get_field_index(dim), dim, table[dim].cast(pa.int32())
                )
        out_path = os.path.join(output_dir, shard_name)
        with open(out_path, "wb") as f:
            writer = pa.ipc.new_stream(f, table.schema)
            writer.write_table(table)
            writer.close()
        print(f"Saved {len(rows)} scored samples to {out_path}")


def main(
    target_model,
    data_dir,
    judge_model,
    dimensions,
    output_dir,
    max_connections=32,
    shards=None,
    display="none",
):
    # Query vllm for the actual served model name to avoid name mismatches
    import requests as rq

    models = rq.get(os.environ["OPENAI_BASE_URL"] + "/models").json()
    judge_model = "openai/" + models["data"][0]["id"]

    # Patch petri's internal get_model call to strip reasoning params that vllm
    # doesn't support (reasoning_effort="high" causes a 400 from the vllm endpoint).
    import petri.scorers.judge as _petri_judge

    _orig_get_model = _petri_judge.get_model

    def _patched_get_model(model=None, *, config=GenerateConfig(), **kwargs):
        if isinstance(model, str) and model.startswith("openai/"):
            config = GenerateConfig(max_tokens=16384)
            # config = GenerateConfig(max_tokens=16384, reasoning_effort="low", temperature=0.0)
        return _orig_get_model(model, config=config, **kwargs)

    _petri_judge.get_model = _patched_get_model

    # Single scorer for all dimensions — avoids N separate judge calls per sample
    scorer = alignment_judge(
        prompt=JUDGE_PROMPT.format(
            target_model=target_model,
            dimensions_text="{dimensions_text}",
            scores_list="{scores_list}",
        ),
        dimensions=dimensions,
        max_score=10,
        model=judge_model,
    )
    conversations = load_conversations_from_shards(data_dir, shards=shards)

    task = Task(
        dataset=[
            Sample(
                input="unused",
                metadata={
                    "conversation_hash": example["conversation_hash"],
                    "generated_conversation": example["generated_conversation"],
                    "_shard": example["_shard"],
                },
            )
            for example in conversations
        ],
        solver=load_conversation(),
        scorer=scorer,
    )
    results = eval(
        task, model=judge_model, max_connections=max_connections, display=display
    )

    save_results(
        results[0],
        dimensions=dimensions,
        output_dir=output_dir,
        conversations=conversations,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", help="Path to the folder of Arrow shards to judge"
    )
    parser.add_argument(
        "--trait", choices=TRAITS, required=True, help="Dimension to judge"
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-connections", type=int, default=256)
    parser.add_argument(
        "--shards",
        nargs="+",
        default=None,
        help="Shard indices or ranges to process (e.g. 0 1 2-5). Default: all.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        default=False,
        help="Enable inspect's rich UI (default: off for batch jobs).",
    )
    args = parser.parse_args()

    print("\nArguments:\n-----------------------------------------------\n", end="\r\n")
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()), end="\r\n")
    print("\n-----------------------------------------------\n", end="\r\n")

    dimensions = {k: v for k, v in DIMENSIONS.items() if k == args.trait}
    output_dir = f"{args.data_dir}--{args.trait}"

    with vllm_server(
        args.judge_model,
        port=JUDGE_PORT,
        extra_args=[
            "--max-num-seqs",
            str(args.max_connections),
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "16384",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens",
            "32768",
            "--tensor-parallel-size",
            str(torch.cuda.device_count()),
        ],
    ) as judge_client:
        main(
            target_model=args.target_model,
            data_dir=args.data_dir,
            output_dir=output_dir,
            judge_model=args.judge_model,
            dimensions=dimensions,
            max_connections=args.max_connections,
            shards=args.shards,
            display="rich" if args.ui else "plain",
        )
