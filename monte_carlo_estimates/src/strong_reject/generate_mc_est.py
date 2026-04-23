import os
import argparse
import json
import gc
import random
import torch
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

from smc.rewards import _sr_harmful_reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of target model (see `conversers.py` for all the available models).",
    )
    parser.add_argument(
        "--jailbreak-dataset",
        default="monte_carlo_estimates/data/strong_reject/jailbreaks.json",
        help="JSON-formatted jailbreak dataset to use.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for generation."
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=10000,
        help="Number of sequences to sample from the model.",
    )
    parser.add_argument(
        "--judge_batch_size",
        type=int,
        default=256,
        help="Number of sequences to sample from the model.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Max batch size for generation."
    )
    parser.add_argument("--output-dir", help="Path to output directory.", required=True)
    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?

    random.seed(42)
    # load the jailbreak dataset
    jailbreaks_dataset = load_dataset("json", data_files=args.jailbreak_dataset)[
        "train"
    ]
    jailbreaks_dataset = jailbreaks_dataset.map(
        lambda example: {
            "jailbreak_prompt_text": example["jailbroken_prompt"][0]["content"]
        }
    )

    print("WARNING: filtering for 'none' jailbreak")
    jailbreaks_dataset = jailbreaks_dataset.filter(lambda x: x["jailbreak"] == "none")

    print("jailbreaks dataset length: ", len(jailbreaks_dataset))

    # set the sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        n=min(args.num_return_sequences, args.batch_size),
    )

    # load the model
    llm = LLM(
        model=args.target_model,
        enable_prefix_caching=True,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=1000,
        gpu_memory_utilization=0.9,
    )

    # generate responses
    convos = [
        [{"role": "user", "content": s}]
        for s in jailbreaks_dataset["jailbreak_prompt_text"]
    ]
    outputs = []
    for _ in range(max(args.num_return_sequences // args.batch_size, 1)):
        outputs.append(llm.chat(convos, sampling_params))

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # dump responses
    output_fname = (
        f"{args.output_dir}/{args.target_model.split('/')[1]}_mc_est_10k.json"
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_fname, "w") as f:
        for zip_tuple in tqdm(zip(jailbreaks_dataset, *outputs)):
            in_data = zip_tuple[0]
            meta_output = list(zip_tuple[1:])
            responses = [r.text for output in meta_output for r in output.outputs]
            in_data.update(
                {
                    "response": responses,
                    "score": _sr_harmful_reward_fn(
                        forbidden_prompt=in_data["forbidden_prompt"],
                        responses=responses,
                    ).tolist(),
                }
            )
            f.write(json.dumps({k: v for k, v in in_data.items()}) + "\n")
