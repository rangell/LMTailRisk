"""Generate responses to WildChat prompts."""

import argparse
import asyncio
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from openai import AsyncOpenAI, BadRequestError
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm_utils import vllm_server


async def call_model(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    temperature: float,
    thinking: bool = True,
) -> tuple[str, str]:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **({"extra_body": {"include_reasoning": True}} if thinking else {}),
        )
    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or ""
    content = msg.content or ""
    if thinking:
        # Qwen3 embeds thinking in <think>...</think> tags within content
        if not reasoning and content.startswith("<think>"):
            end = content.find("</think>")
            if end != -1:
                reasoning = content[7:end].strip()
                content = content[end + 8 :].strip()
            else:
                # thinking block never closed (truncated); treat all as thinking
                reasoning = content[7:].strip()
                content = ""
    return reasoning, content


def estimate_tokens(messages: list[dict], tokenizer: AutoTokenizer) -> int:
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return len(tokens)


def extract_prompt_messages(conversation: list[dict]) -> list[dict]:
    """Return the system prompt (if any) and the first user turn."""
    messages = [{"role": m["role"], "content": m["content"]} for m in conversation]
    result = []
    # include leading system prompt if present
    if messages and messages[0]["role"] == "system":
        result.append(messages[0])
    # find first user turn
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if first_user is None:
        return []
    result.append(messages[first_user])
    return result


GEN_CONV_SCHEMA = pa.list_(
    pa.struct(
        [
            ("role", pa.string()),
            ("thinking", pa.string()),
            ("content", pa.string()),
        ]
    )
)


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


def find_shards(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("data-*.arrow"))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses for WildChat shards"
    )
    parser.add_argument(
        "--data-dir",
        default="data/allenai_WildChat_filtered__english_only+first_prompt_le_512_tokens/",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--shards",
        nargs="+",
        default=None,
        help="Shard indices or ranges to process (e.g. 0 1 2-5). Default: all.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help="Disable thinking/reasoning extraction (for non-thinking models)",
    )
    args = parser.parse_args()

    short_model_name = args.model.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = f"data/{short_model_name}_responses"

    print("\nArguments:\n-----------------------------------------------\n", end="\r\n")
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()), end="\r\n")
    print("\n-----------------------------------------------\n", end="\r\n")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_shards = find_shards(data_dir)
    if not all_shards:
        raise FileNotFoundError(f"No data-*.arrow files found in {data_dir}")

    shard_indices = (
        parse_shards(args.shards, len(all_shards))
        if args.shards
        else list(range(len(all_shards)))
    )
    shard_paths = [all_shards[i] for i in shard_indices]

    print(f"Processing {len(shard_paths)} shard(s): {[p.name for p in shard_paths]}")

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with vllm_server(
        args.model,
        extra_args=[
            "--max-model-len",
            str(args.max_model_len),
        ],
    ) as client:
        for shard_path in shard_paths:
            out_path = output_dir / shard_path.name
            if out_path.exists():
                print(f"Skipping {shard_path.name} (output already exists)")
                continue

            print(f"\nLoading {shard_path.name}...")
            reader = ipc.open_stream(shard_path)
            table = reader.read_all()
            rows = table.to_pylist()

            print(f"  {len(rows)} rows — generating responses...")

            async def run_shard(rows):
                semaphore = asyncio.Semaphore(args.concurrency)
                results = [None] * len(rows)

                with tqdm(total=len(rows), unit="sample") as pbar:

                    async def process_one(idx, row):
                        msgs = extract_prompt_messages(row["conversation"])
                        if not msgs:
                            pbar.update(1)
                            return
                        try:
                            thinking, content = await call_model(
                                client,
                                args.model,
                                msgs,
                                semaphore,
                                temperature=args.temperature,
                                thinking=not args.no_thinking,
                            )
                        except BadRequestError:
                            pbar.update(1)
                            return
                        results[idx] = msgs + [
                            {
                                "role": "assistant",
                                "thinking": thinking,
                                "content": content,
                            }
                        ]
                        pbar.update(1)

                    await asyncio.gather(
                        *[process_one(i, row) for i, row in enumerate(rows)]
                    )

                return results

            gen_convs = asyncio.run(run_shard(rows))

            new_table = table.append_column(
                pa.field("generated_conversation", GEN_CONV_SCHEMA),
                pa.array(gen_convs, type=GEN_CONV_SCHEMA),
            )

            with ipc.new_stream(out_path, new_table.schema) as writer:
                writer.write_table(new_table)

            print(f"  Written to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
