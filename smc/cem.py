import os
import pickle
import torch
import numpy as np

from smc.utils import custom_print, wait_for_everyone, is_main_process


def run_cem(
    estimator,
    example,
    outputs,
    steering_params_arr,
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

    custom_print(f"Length of completions: {len(completions)}")
    custom_print(f"Length of completion ids: {len(completion_ids)}")
    custom_print(f"Length of full input ids: {len(full_input_ids)}")

    if len(completions) == 0:
        custom_print("WARNING: All generations were filtered out...")
        custom_print("Returning 0.0 for CEM estimate")
        return 0.0

    cross_entropy_tensor = estimator.estimate_CEM_harmful_trait(
        prompt=example["forbidden_prompt"],
        completions=completions,
        completion_ids=completion_ids,
        full_input_ids=full_input_ids,
        judge_scores=judge_scores,
        importance_weights=importance_weights,
        steering_params_arr=steering_params_arr,
        proposal_idx_switch_arr=proposal_idx_switch_arr,
        proposal_bias_arr=proposal_bias_arr,
    )
    if cross_entropy_tensor is not None:
        argmin_indices = torch.unravel_index(
            torch.argmin(cross_entropy_tensor), cross_entropy_tensor.shape
        )
        custom_print(
            f"Optimal steering_params: {steering_params_arr[argmin_indices[0]]},"
            f"Optimal proposal_idx_switch: {proposal_idx_switch_arr[argmin_indices[1]]},"
            f"Optimal proposal_bias: {proposal_bias_arr[argmin_indices[2]]}",
        )

        _, _out = estimator.estimate_harmful_trait(
            prompt=example["forbidden_prompt"],
            steering_params=steering_params_arr[argmin_indices[0]],
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
    steering_params_arr,
    proposal_idx_switch_arr,
    proposal_bias_arr,
):
    """Sets the best proposal hyperparameters in args using CEM."""
    cross_entropy_tensor = 0

    model_output_dict = {}
    for prompt_idx, example in enumerate(mc_dataset):
        model_output_dict[prompt_idx] = {}
        model_output_dict[prompt_idx]["forbidden_prompt"] = example["forbidden_prompt"]

        custom_print(
            f"Forbidden prompt ({prompt_idx + 1}/{len(mc_dataset)}): {example['forbidden_prompt']}"
        )
        custom_print(f"Monte Carlo harm estimate: {float(example['harm_mean'])}")

        model_output_dict[prompt_idx]["mc_scores"] = example["harm_scores"]
        model_output_dict[prompt_idx]["mc_mean"] = float(example["harm_mean"])

        # custom_print(f"Monte Carlo harm estimate: {float(np.mean(example['score']))}")

        # model_output_dict[prompt_idx]["mc_scores"] = example["score"]
        # model_output_dict[prompt_idx]["mc_mean"] = np.mean(example["score"])

        # _, outputs = estimator.estimate_harmful_trait(
        #    prompt=example["forbidden_prompt"],
        #    steering_params=args.steering_params,
        #    # proposal_bias=args.proposal_bias,
        #    # proposal_idx_switch=args.proposal_idx_switch,
        # )

        outputs = {}
        for sub_idx, steering_params in enumerate(steering_params_arr):
            wait_for_everyone()
            custom_print(f"Steering params: {steering_params}")
            custom_print(
                f"Forbidden prompt ({prompt_idx + 1}/{len(mc_dataset)}): {example['forbidden_prompt']}",
            )
            # custom_print(f"Monte Carlo harm estimate: {float(np.mean(example['score']))}", )
            #
            custom_print(f"Monte Carlo harm estimate: {float(example['harm_mean'])}")
            _, _out = estimator.estimate_harmful_trait(
                prompt=example["forbidden_prompt"], steering_params=steering_params
            )

            custom_print(
                "\n-----------------------------------------------\n",
            )

            outputs = merge_dicts(outputs, _out)

        cross_entropy_tensor += run_cem(
            estimator,
            example,
            outputs,
            steering_params_arr,
            proposal_idx_switch_arr,
            proposal_bias_arr,
        )

        # custom_print(f"cross_entropy_tensor: {cross_entropy_tensor}")

        for key in outputs:
            model_output_dict[prompt_idx][key] = outputs[key]

        if prompt_idx % 5 == 0 and is_main_process():
            with open(os.path.join(output_dir, "cem_model_outputs.pkl"), "wb") as f:
                pickle.dump(model_output_dict, f)
        wait_for_everyone()

    if is_main_process():
        with open(os.path.join(output_dir, "cem_model_outputs.pkl"), "wb") as f:
            pickle.dump(model_output_dict, f)
    wait_for_everyone()

    argmin_indices = torch.unravel_index(
        torch.argmin(cross_entropy_tensor), cross_entropy_tensor.shape
    )
    args.steering_params = steering_params_arr[argmin_indices[0]]
    args.proposal_idx_switch = proposal_idx_switch_arr[argmin_indices[1]]
    args.proposal_bias = proposal_bias_arr[argmin_indices[2]]
    custom_print(
        f"Optimal steering_params: {steering_params_arr[argmin_indices[0]]},"
        f"Optimal proposal_idx_switch: {proposal_idx_switch_arr[argmin_indices[1]]},"
        f"Optimal proposal_bias: {proposal_bias_arr[argmin_indices[2]]}",
    )
