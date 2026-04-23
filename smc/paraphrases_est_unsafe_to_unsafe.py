from typing import List, ContextManager, Callable
import argparse
import json
from pathlib import Path
import os
import datetime
import random
import time
import pickle

from datasets import load_dataset
import numpy as np
import torch

from refusal_direction.pipeline.utils.hook_utils import add_hooks

from smc.model_utils import (
    load_model_and_tokenizer,
    load_refusal_direction,
    get_all_direction_ablation_hooks,
)
from smc.rewards import _sr_harmful_reward_fn
from smc.estimator import HarmfulTraitEstimator


class UnsafeToUnsafeEstimator(HarmfulTraitEstimator):
    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)

        # Load refusal direction for proposal model
        refusal_direction = load_refusal_direction(args.refusal_direction_path)

        # Construct torch hooks for ablating refusal
        self.ablation_fwd_pre_hooks = {}
        self.ablation_fwd_hooks = {}
        for ablation_intensity in self.args.ablation_intensities:
            (
                self.ablation_fwd_pre_hooks[str(ablation_intensity)],
                self.ablation_fwd_hooks[str(ablation_intensity)],
            ) = get_all_direction_ablation_hooks(
                model,
                refusal_direction["direction"],
                ablation_intensity=ablation_intensity,
            )

    def proposal_context_manager(self, ablation_intensity: float) -> ContextManager:
        return add_hooks(
            module_forward_pre_hooks=self.ablation_fwd_pre_hooks[
                str(ablation_intensity)
            ],
            module_forward_hooks=self.ablation_fwd_hooks[str(ablation_intensity)],
        )

    def judge_responses(self, prompt: str, responses: List[str]) -> torch.Tensor:
        scores = _sr_harmful_reward_fn(
            forbidden_prompt=prompt,
            responses=responses,
        )
        return scores

    def create_reward_function(self, prompt: str) -> Callable:
        def reward_fn(input_ids):
            scores = _sr_harmful_reward_fn(
                forbidden_prompt=prompt,
                responses=self.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                ),
            )
            return scores

        return reward_fn


def run_cem(
    estimator,
    example,
    outputs,
    steering_coef_arr,
    proposal_idx_switch_arr,
    proposal_bias_arr,
):
    keep_mask = outputs["judge_scores"] > 0.0
    keep_mask &= outputs["importance_weights"].squeeze() > 0.0

    judge_scores = outputs["judge_scores"][keep_mask]
    importance_weights = outputs["importance_weights"][keep_mask]

    completions, completion_ids, full_input_ids = [], [], []
    for i, keep in enumerate(keep_mask):
        if keep:
            completions.append(outputs["responses"][i])
            full_input_ids.append(outputs["_input_ids"][i])
            completion_ids.append(outputs["_completion_ids"][i])

    print(f"Length of completions: {len(completions)}")
    print(f"Length of completion ids: {len(completion_ids)}")
    print(f"Length of full input ids: {len(full_input_ids)}")

    if len(completions) == 0:
        print("WARNING: All generations were filtered out...")
        print("Returning 0.0 for CEM estimate")
        return 0.0

    cross_entropy_tensor = estimator.estimate_CEM_harmful_trait(
        prompt=example["forbidden_prompt"],
        completions=completions,
        completion_ids=completion_ids,
        full_input_ids=full_input_ids,
        judge_scores=judge_scores,
        importance_weights=importance_weights,
        steering_coef_arr=steering_coef_arr,
        proposal_idx_switch_arr=proposal_idx_switch_arr,
        proposal_bias_arr=proposal_bias_arr,
    )
    if cross_entropy_tensor is not None:
        argmin_indices = torch.unravel_index(
            torch.argmin(cross_entropy_tensor), cross_entropy_tensor.shape
        )
        print(
            f"Optimal ablation_intensity: {steering_coef_arr[argmin_indices[0]]}, Optimal proposal_idx_switch: {proposal_idx_switch_arr[argmin_indices[1]]}, Optimal proposal_bias: {proposal_bias_arr[argmin_indices[2]]}"
        )

        _, _out = estimator.estimate_harmful_trait(
            prompt=example["forbidden_prompt"],
            steering_coef=steering_coef_arr[argmin_indices[0]],
            proposal_idx_switch=proposal_idx_switch_arr[argmin_indices[1]],
            proposal_bias=proposal_bias_arr[argmin_indices[2]],
        )

        return cross_entropy_tensor / cross_entropy_tensor.sum()
    else:
        return 0
        


def merge_dicts(sink_dict, source_dict):
    if not sink_dict:
        new_sink_dict = source_dict
    else:
        new_sink_dict = {}
        for k, v in sink_dict.items():
            assert k in source_dict.keys()
            if isinstance(v, list):
                new_sink_dict[k] = sink_dict[k] + source_dict[k]
            elif isinstance(v, np.ndarray):
                new_sink_dict[k] = np.concatenate(
                    (sink_dict[k], source_dict[k]), axis=0
                )

    return new_sink_dict


def set_proposal_hparams(
    args,
    output_dir,
    mc_dataset,
    estimator,
    steering_coef_arr,
    proposal_idx_switch_arr,
    proposal_bias_arr,
):
    """Sets the best proposal hyperparameters in args using CEM."""
    cross_entropy_tensor = 0

    model_output_dict = {}
    for prompt_idx, example in enumerate(mc_dataset):
        model_output_dict[prompt_idx] = {}
        model_output_dict[prompt_idx]["forbidden_prompt"] = example["forbidden_prompt"]

        print(
            f"Forbidden prompt ({prompt_idx}/{len(mc_dataset) - 1}): {example['forbidden_prompt']}"
        )
        print(f"Monte Carlo harm estimate: {float(np.mean(example['score']))}")

        model_output_dict[prompt_idx]["mc_scores"] = example["score"]
        model_output_dict[prompt_idx]["mc_mean"] = np.mean(example["score"])

        outputs = {}

        for sub_idx, ablation_intensity in enumerate(args.ablation_intensities):
            print(f"Ablation intensity: {ablation_intensity}")
            print(
                f"Forbidden prompt ({prompt_idx}/{len(mc_dataset) - 1}): {example['forbidden_prompt']}"
            )
            print(f"Monte Carlo harm estimate: {float(np.mean(example['score']))}")

            _, _out = estimator.estimate_harmful_trait(
                prompt=example["forbidden_prompt"],
                steering_coef=ablation_intensity,
            )

            print("\n-----------------------------------------------\n")

            outputs = merge_dicts(outputs, _out)

        cross_entropy_tensor += run_cem(
            estimator,
            example,
            outputs,
            steering_coef_arr,
            proposal_idx_switch_arr,
            proposal_bias_arr,
        )

        for key in outputs:
            model_output_dict[prompt_idx][key] = outputs[key]

        if prompt_idx % 5 == 0:
            with open(os.path.join(output_dir, "cem_model_outputs.pkl"), "wb") as f:
                pickle.dump(model_output_dict, f)

    with open(os.path.join(output_dir, "cem_model_outputs.pkl"), "wb") as f:
        pickle.dump(model_output_dict, f)

    argmin_indices = torch.unravel_index(
        torch.argmin(cross_entropy_tensor), cross_entropy_tensor.shape
    )
    args.ablation_intensity = steering_coef_arr[argmin_indices[0]]
    args.proposal_idx_switch = proposal_idx_switch_arr[argmin_indices[1]]
    args.proposal_bias = proposal_bias_arr[argmin_indices[2]]
    print(
        f"Optimal ablation_intensity: {steering_coef_arr[argmin_indices[0]]}, Optimal proposal_idx_switch: {proposal_idx_switch_arr[argmin_indices[1]]}, Optimal proposal_bias: {proposal_bias_arr[argmin_indices[2]]}"
    )


def main(args):
    args.model_shortname = args.model_name.split("/")[1]
    args.refusal_direction_path = (
        f"refusal_direction/pipeline/runs/{args.model_shortname}/"
    )

    # if args.mc_est_dataset == "":
    #     args.mc_est_dataset = f"monte_carlo_estimates/results/strong_reject/{args.model_shortname}_mc_est_10k.json"
        
    if args.mc_est_dataset == "":
        args.mc_est_dataset = f"3_combined_paraphrased_results_with_mc_{args.model_shortname}.json"

    if args.proposal_idx_switch == -1:
        args.proposal_idx_switch = args.max_new_tokens + 1

    print("\nArguments:\n-----------------------------------------------\n")
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print("\n-----------------------------------------------\n")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / args.model_shortname / timestamp

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        print(f"Output directory {output_dir} already exists.")
        # wait for a random time to avoid overwriting
        wait_time = random.randint(1, 200)
        print(f"Waiting for {wait_time} seconds before proceeding...")
        time.sleep(wait_time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir
        output_dir = Path(output_dir) / args.model_shortname / timestamp
        os.makedirs(output_dir, exist_ok=False)

    metadata = vars(args)
    metadata["timestamp"] = timestamp
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Experiment metadata saved to {metadata_file}")

    # For reproducability
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model and tokenizer from huggingface
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    estimator = UnsafeToUnsafeEstimator(
        args=args,
        model=model,
        tokenizer=tokenizer,
    )

    if args.use_cem:
        # This will set the args for proposal model hyperparameters
        mc_dataset = load_dataset(
            "json",
            data_files=args.mc_est_dataset,
        )["train"]
        mc_dataset = mc_dataset.select(range(min(int(args.cem_frac * len(mc_dataset)), 2)))

        steering_coef_arr = [0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
        proposal_idx_switch_arr = list(range(0, 151, 5))
        proposal_bias_arr = np.linspace(0, 1, 10)
        set_proposal_hparams(
            args,
            output_dir,
            mc_dataset,
            estimator,
            steering_coef_arr,
            proposal_idx_switch_arr,
            proposal_bias_arr,
        )

    # Load judged Monte Carlo samples
    mc_dataset = load_dataset(
        "json",
        data_files=args.mc_est_dataset,
    )["train"]

    model_output_dict = {}
    for prompt_idx, example in enumerate(mc_dataset):
        model_output_dict[prompt_idx] = {}
        model_output_dict[prompt_idx]["forbidden_prompt"] = example["forbidden_prompt"]

        print(
            f"Forbidden prompt ({prompt_idx}/{len(mc_dataset) - 1}): {example['forbidden_prompt']}"
        )
        print(f"Monte Carlo harm estimate: {float(np.mean(example['score']))}")

        model_output_dict[prompt_idx]["mc_scores"] = example["score"]
        model_output_dict[prompt_idx]["mc_mean"] = np.mean(example["score"])
        if "original_forbidden_prompt" in example:
            model_output_dict[prompt_idx]["original_forbidden_prompt"] = example["original_forbidden_prompt"]

        _, outputs = estimator.estimate_harmful_trait(
            prompt=example["forbidden_prompt"],
            steering_coef=args.ablation_intensity,
            proposal_idx_switch=args.proposal_idx_switch,
            proposal_bias=args.proposal_bias,
        )

        print("\n-----------------------------------------------\n")

        for key in outputs:
            model_output_dict[prompt_idx][key] = outputs[key]

        if prompt_idx % 5 == 0:
            with open(os.path.join(output_dir, "model_outputs.pkl"), "wb") as f:
                pickle.dump(model_output_dict, f)

    with open(os.path.join(output_dir, "model_outputs.pkl"), "wb") as f:
        pickle.dump(model_output_dict, f)


def get_args():
    # NOTE: Currently all unused arguments are commented out
    parser = argparse.ArgumentParser()

    add_arguments(parser)

    args = parser.parse_args()

    return args


def add_arguments(parser):
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_output",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model which we want to estimate harmfulness.",
    )
    parser.add_argument(
        "--mc_est_dataset",
        type=str,
        default="",
        help="JSON file containing dataset of monte carlo estimates.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=100,
        help="Number of particles for each generation step.",
    )
    parser.add_argument(
        "--fwd_batch_size",
        type=int,
        default=500,
        help="Batch size for forward pass.",
    )
    parser.add_argument(
        "--proposal_bias",
        type=float,
        default=1.0,
        help="Interpolation factor (in [0, 1] -> 0 means base model proposal -> 1 means full ablation proposal).",
    )
    parser.add_argument(
        "--proposal_idx_switch",
        type=int,
        default=-1,
        help="When to switch from ablated model to target model as proposal.",
    )
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=32,
        help="Batch size for reward function.",
    )
    parser.add_argument(
        "--n_judge_jobs",
        type=int,
        default=1000,
        help="Number of parallel processes for judge.",
    )
    parser.add_argument(
        "--use_smc",
        action="store_true",
        help="Whether to use Sequential Monte Carlo (SMC) for generation.",
    )
    parser.add_argument(
        "--low_vram_cache",
        action="store_true",
        help="Whether to use Sequential Monte Carlo (SMC) for generation.",
    )
    parser.add_argument(
        "--use_importance_weights_in_resampling",
        action="store_true",
        help="Whether to use importance weights in resampling for SMC.",
    )
    parser.add_argument(
        "--smc_verbose", action="store_true", help="Whether to print SMC logs"
    )
    parser.add_argument(
        "--use_cem", action="store_true", help="Use CEM to select proposal model."
    )
    parser.add_argument(
        "--cem_frac", type=float, default=0.005, help="Fraction of data to use for CEM."
    )

    ################### SPECIFIC ARGS #############################

    parser.add_argument(
        "--ablation_intensity",
        type=float,
        default=1.0,
        help="Fraction of the refusal direction to ablate in proposal (in [0, 1]).",
    )

    parser.add_argument(
        "--ablation_intensities",
        type=float,
        nargs="+",  # Expects one or more space-separated floats
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Available set of ablation intensities.",
    )

    ###############################################################


if __name__ == "__main__":
    args = get_args()
    main(args)
