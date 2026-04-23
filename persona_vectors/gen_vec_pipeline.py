import os
import gc
import json
import torch
import subprocess
import logging
import fire
from itertools import product

from transformers import AutoConfig

import nest_asyncio

from persona_vectors.eval.eval_persona import main as eval_persona_extract
from persona_vectors.generate_vec import save_persona_vector

nest_asyncio.apply()

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main(model, trait, judge_model="openai/gpt-oss-120b"):
    model_shortname = model.split("/")[1]

    # Set refusal_ablation and ablation_intensity here!
    ablate_refusal = True
    ablation_intensity = 0.8

    persona_instruction_type = "pos"
    pos_output_path = f"persona_vectors/eval_persona_extract/{model_shortname}/{trait}_{persona_instruction_type}_instruct.csv"
    assistant_name = trait
    eval_persona_extract(
        model,
        trait,
        pos_output_path,
        ablate_refusal=ablate_refusal,
        ablation_intensity=ablation_intensity,
        max_tokens=1000,
        n_per_question=10,
        batch_process=True,
        max_concurrent_judges=100,
        persona_instruction_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version="extract",
        overwrite=True,
        judge_launch_script="launch_judge_local.sh",
    )

    gc.collect()
    torch.cuda.empty_cache()

    persona_instruction_type = "neg"
    neg_output_path = f"persona_vectors/eval_persona_extract/{model_shortname}/{trait}_{persona_instruction_type}_instruct.csv"
    assistant_name = "helpful"
    eval_persona_extract(
        model,
        trait,
        neg_output_path,
        ablate_refusal=False,
        ablation_intensity=0.0,
        max_tokens=1000,
        n_per_question=10,
        batch_process=True,
        max_concurrent_judges=100,
        persona_instruction_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version="extract",
        overwrite=True,
        judge_launch_script="launch_judge_local.sh",
    )

    gc.collect()
    torch.cuda.empty_cache()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    busy_gpu_subprocess = subprocess.Popen(["python", "misc/run_50xALL.py"])

    save_dir = f"persona_vectors/persona_vectors/{model_shortname}/"
    save_persona_vector(
        model,
        pos_output_path,
        neg_output_path,
        trait,
        save_dir,
        ablate_refusal=ablate_refusal,
        ablation_intensity=ablation_intensity,
    )

    busy_gpu_subprocess.terminate()
    busy_gpu_subprocess.wait()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

    gc.collect()
    torch.cuda.empty_cache()

    model_config = AutoConfig.from_pretrained(model)

    coefs = [1.0, 1.5, 2.0, 2.5, 3.0]
    layers = [
        int(model_config.num_hidden_layers * layer_frac)
        for layer_frac in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    ]

    best_success_rate = 0.0
    best_idx = 0
    best_coef = None
    best_layer = None

    for idx, (coef, layer) in enumerate(product(coefs, layers)):
        output_path = f"persona_vectors/eval_persona_eval/{trait}/{model_shortname}-steering_results-{idx}.csv"
        success_rate = eval_persona_extract(
            model,
            trait,
            output_path=output_path,
            ablate_refusal=False,
            ablation_intensity=0.0,
            max_tokens=1000,
            n_per_question=10,
            batch_process=True,
            max_concurrent_judges=100,
            assistant_name=assistant_name,
            judge_model=judge_model,
            version="eval",
            overwrite=True,
            judge_launch_script="launch_judge_local.sh",
            coef=coef,
            layer=layer,
            steering_type="response",
            vector_path=f"persona_vectors/persona_vectors/{model_shortname}/{trait}_response_avg_diff.pt",
        )

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_idx = idx
            best_coef = coef
            best_layer = layer

    steering_stats = {
        "best_success_rate": best_success_rate,
        "best_idx": best_idx,
        "best_coef": best_coef,
        "best_layer": best_layer,
    }

    print("Best steering stats: ", json.dumps(steering_stats, indent=4))

    with open(
        f"persona_vectors/persona_vectors/{model_shortname}/{trait}_best_steering_stats.json",
        "w",
    ) as f:
        json.dump(steering_stats, f)


if __name__ == "__main__":
    fire.Fire(main)
