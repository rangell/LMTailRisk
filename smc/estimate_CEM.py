import argparse
import json
from pathlib import Path
import random
from tqdm.auto import tqdm
import pickle

from datasets import load_dataset
import numpy as np
import torch

from smc.model_utils import load_model_and_tokenizer
from smc.est_unsafe_to_unsafe import UnsafeToUnsafeEstimator

from smc.model_utils import (
    load_model_and_tokenizer,
)


def main(args):
    args.model_shortname = args.model_name.split("/")[1]
    args.refusal_direction_path = (
        f"refusal_direction/pipeline/runs/{args.model_shortname}/"
    )

    if args.mc_est_dataset == "":
        args.mc_est_dataset = f"monte_carlo_estimates/results/strong_reject/{args.model_shortname}_mc_est_10k.json"


    print("\nArguments:\n-----------------------------------------------\n")
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print("\n-----------------------------------------------\n")

    # output_dir = args.output_dir
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = Path(output_dir) / args.model_shortname / timestamp

    # try:
    #     os.makedirs(output_dir)
    # except FileExistsError:
    #     print(f"Output directory {output_dir} already exists.")
    #     # wait for a random time to avoid overwriting
    #     wait_time = random.randint(1, 200)
    #     print(f"Waiting for {wait_time} seconds before proceeding...")
    #     time.sleep(wait_time)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_dir = args.output_dir
    #     output_dir = Path(output_dir) / args.model_shortname / timestamp
    #     os.makedirs(output_dir, exist_ok=False)

    # metadata = vars(args)
    # metadata["timestamp"] = timestamp
    # metadata_file = os.path.join(output_dir, "metadata.json")
    # with open(metadata_file, "w") as f:
    #     json.dump(metadata, f, indent=4)
    # print(f"Experiment metadata saved to {metadata_file}")

    # For reproducability
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model and tokenizer from huggingface
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Load judged monte carlo samples
    mc_dataset = load_dataset(
        "json",
        data_files=args.mc_est_dataset,
    )["train"]

    estimator = UnsafeToUnsafeEstimator(
        args=args,
        model=model,
        tokenizer=tokenizer,
    )

    cem_model_output_path = Path(
      "/gpfs/data/ranganathlab/singhr36/rare-generations/model_output/Llama-3.2-1B-Instruct/20251205_142551"
    )

    cem_model_output_dict = cem_model_output_path / "model_outputs.pkl"
    cem_metadata = cem_model_output_path / "metadata.json"

    with open(cem_model_output_dict, "rb") as f:
        cem_model_output_dict = pickle.load(f)

    with open(cem_metadata, "r") as f:
        cem_metadata = json.load(f)

    model_output_dict = {}
    
    cross_entropy_matrix = 0
    proposal_idx_switch_arr = [0, 10, 20, 50, 100, -1]
    proposal_bias_arr = [0, 0.25, 0.5, 0.75, 1]
    # for prompt_idx in tqdm(range(len(cem_model_output_dict))):

    for prompt_idx in tqdm(range(100)):
        example = cem_model_output_dict[prompt_idx]

        model_output_dict[prompt_idx] = {}
        model_output_dict[prompt_idx]["forbidden_prompt"] = example["forbidden_prompt"]

        # print(f"Forbidden prompt: {example['forbidden_prompt']}")
        # print(f"Monte Carlo harm estimate: {float(np.mean(example['judge_scores']))}")

        model_output_dict[prompt_idx]["mc_scores"] = example["judge_scores"]
        model_output_dict[prompt_idx]["mc_mean"] = np.mean(example["judge_scores"])
        
        cross_entropy_matrix += estimator.estimate_CEM_harmful_trait(
            prompt=example["forbidden_prompt"],
            completions=example["responses"],
            completion_ids=example['_completion_ids'],
            full_input_ids=example["_input_ids"],
            judge_scores=example["judge_scores"],
            importance_weights=example["importance_weights"],
            proposal_idx_switch_arr=proposal_idx_switch_arr,
            proposal_bias_arr=proposal_bias_arr
        )
        
        # print("\n-----------------------------------------------\n")
        # print(cross_entropy_matrix / (prompt_idx + 1))
        
    cross_entropy_matrix /= len(cem_model_output_dict)
    print("Cross-entropy matrix (rows: proposal_idx_switch, cols: base_mixing):")
    print(cross_entropy_matrix)
    
    argmin_indices = torch.unravel_index(torch.argmin(cross_entropy_matrix), cross_entropy_matrix.shape)
    print(f"Optimal proposal_idx_switch: {proposal_idx_switch_arr[argmin_indices[0]]}, Optimal proposal_bias: {proposal_bias_arr[argmin_indices[1]]}")
    
    minimal_cross_entropy = cross_entropy_matrix[argmin_indices[0], argmin_indices[1]]
    print(f"Minimal cross-entropy: {minimal_cross_entropy}")
    
    return cross_entropy_matrix, minimal_cross_entropy, proposal_idx_switch_arr[argmin_indices[0]], proposal_bias_arr[argmin_indices[1]]


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

    ###############################################################


if __name__ == "__main__":
    args = get_args()
    
    data_gen = {}
    cross_ent_matrix_dict = {}
    
    for ablation_intensity in [0.25, 0.5, 0.75, 1.0]:
        args.ablation_intensity = ablation_intensity
        print(f"\n\nRunning CEM estimation with ablation intensity: {ablation_intensity}\n\n")
        cross_ent_matrix, minimal_cross_entropy, switch_idx, bias_idx = main(args)
        
        data_gen[ablation_intensity] = {
            # "cross_entropy_matrix": cross_ent_matrix.tolist(),
            "minimal_cross_entropy": minimal_cross_entropy,
            "optimal_proposal_idx_switch": switch_idx,
            "optimal_proposal_bias": bias_idx
        }
        cross_ent_matrix_dict[ablation_intensity] = cross_ent_matrix.tolist()
        
    print("\n\nFinal results across ablation intensities:\n")    
    print(data_gen)
    
    print("\n\nCross-entropy matrices across ablation intensities:\n")
    print(cross_ent_matrix_dict)
