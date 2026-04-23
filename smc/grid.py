import argparse
import yaml
from pathlib import Path
import os
import itertools
from tqdm import tqdm

# from est_unsafe_to_unsafe import main as launch_experiment


def create_grid():
    grid = {
        # "output_dir": "output_paraphrases_est_unsafe_to_unsafe_3",
        "output_dir": "icml_unsafe_seed_44",
        "model_name": [
            # "meta-llama/llama-2-7b-chat-hf",
            "meta-llama/Llama-3.1-8B-Instruct",
            # "microsoft/phi-4",
            # "meta-llama/Llama-3.2-1B-Instruct",
            # "allenai/Olmo-3-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            # "google/gemma-2-9b-it",
            # "Qwen/Qwen2.5-32B-Instruct",
        ],
        "num_particles": [500],
        "seed": 44,
        # "mc_est_dataset": ["advbench_mc.json",],
    }

    return grid


def generate_args(index):
    grid_dict = create_grid()

    hp_grid = []

    # Separate list parameters from scalar parameters
    list_params = {k: v for k, v in grid_dict.items() if isinstance(v, list)}
    scalar_params = {k: v for k, v in grid_dict.items() if not isinstance(v, list)}

    param_names = list(list_params.keys())
    param_values = list(list_params.values())

    # Generate all combinations using itertools.product
    all_combinations = list(itertools.product(*param_values))

    for combo in tqdm(all_combinations):
        # Create parameter dictionary for this combination
        params = dict(zip(param_names, combo))
        params.update(scalar_params)

        if params not in hp_grid:
            hp_grid.append(params)

    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)

    new_args = hp_grid[index]
    # config = {}

    # config = argparse.Namespace(**config)
    print(f"len(hp_grid): {len(hp_grid)}")
    print(f"Selecting {index+1} out of {len(hp_grid)} configurations")

    # Update args with the new configuration
    config = {}
    for key, value in new_args.items():
        # setattr(config, key, value)
        config[key] = value

    config = argparse.Namespace(**config)
    print(config)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    config = generate_args(args.index)
    
    # run importance sampling estimator
    exp_command = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --module smc.est_unsafe_to_unsafe --output_dir {} --model_name {}  --num_particles {} --seed {} --fwd_batch_size 500 --max_new_tokens 150 --use_cem"
    exp_command = exp_command.format(config.output_dir, config.model_name, config.num_particles, config.seed)
    
    # # run importance sampling estimator with adversarial benchmark prompts
    # exp_command = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --module smc.paraphrases_est_unsafe_to_unsafe --output_dir {} --model_name {}  --num_particles {} --seed {} --fwd_batch_size 500 --max_new_tokens 150 --use_cem --mc_est_dataset {}"
    # exp_command = exp_command.format(config.output_dir, config.model_name, config.num_particles, config.seed, config.mc_est_dataset)
    
    
    # # run MC estimator
    # exp_command = "uv run --module monte_carlo_estimates.src.strong_reject.generate_mc_est --target-model {} --num-return-sequences {} --output-dir {} --temperature=0.0"    
    # exp_command = exp_command.format(config.model_name, config.num_particles, config.output_dir)
    
    # model_shortname = config.model_name.split("/")[-1]
    # config.jailbreak_dataset = f"2_combined_paraphrased_results_with_mc_{model_shortname}.json"
    # assert Path(config.jailbreak_dataset).exists(), f"Jailbreak dataset {config.jailbreak_dataset} does not exist. Please run smc/create_fake_mc.py to create the combined paraphrased results with MC estimates file before running this experiment."

    # # run paraphrase IS estimator
    # exp_command = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --module smc.paraphrases_est_unsafe_to_unsafe --output_dir {} --model_name {}  --num_particles {} --seed {} --fwd_batch_size 500 --max_new_tokens 150 --use_cem"
    # exp_command = exp_command.format(config.output_dir, config.model_name, config.num_particles, config.seed)
    
    # # run paraphrase MC estimator
    # exp_command = "uv run --module monte_carlo_estimates.src.strong_reject.generate_paraphrase_mc_est --target-model {} --num-return-sequences 1 --output-dir {} --temperature=0.0 --jailbreak-dataset {}"
    # exp_command = exp_command.format(config.model_name, config.output_dir, config.jailbreak_dataset)
        
    
    os.system(
        exp_command
    )
