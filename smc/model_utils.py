import os
import json
import weakref

import torch
from torch import Tensor
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from refusal_direction.pipeline.utils.hook_utils import (
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)

from smc.utils import is_distributed


def load_model_and_tokenizer(model_name_or_path):
    # NOTE: returns model in `eval` mode
    config = AutoConfig.from_pretrained(
        model_name_or_path, output_hidden_states=True, return_dict_in_generate=True
    )

    if isinstance(config, Qwen2Config):
        config = None

    if is_distributed():
        assert False
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            tp_plan="auto",
            token=os.getenv("HF_TOKEN"),
            config=config,
            trust_remote_code=True,
        ).eval()
        print(model._tp_plan)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
            config=config,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        token=os.getenv("HF_TOKEN"),
        padding_side="left",
    )

    # Weird quirk with GraySwanAI model
    if model_name_or_path == "GraySwanAI/Llama-3-8B-Instruct-RR":
        tokenizer.eos_token = "<|eot_id|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_refusal_direction(refusal_direction_path):
    direction_path = os.path.join(refusal_direction_path, "direction.pt")
    metadata_path = os.path.join(refusal_direction_path, "direction_metadata.json")
    direction = torch.load(direction_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata["direction"] = direction
    return metadata


def get_all_direction_ablation_hooks(
    model,
    direction: Tensor,
    ablation_intensity: float = 1.0,
    batch_indices=None,
):
    model_block_modules = weakref.proxy(model.model.layers)
    model_attn_modules = torch.nn.ModuleList(
        [weakref.proxy(block_module.self_attn) for block_module in model_block_modules]
    )
    model_mlp_modules = torch.nn.ModuleList(
        [weakref.proxy(block_module.mlp) for block_module in model_block_modules]
    )

    fwd_pre_hooks = [
        (
            model_block_modules[layer],
            get_direction_ablation_input_pre_hook(
                direction=direction,
                ablation_intensity=ablation_intensity,
                batch_indices=batch_indices,
            ),
        )
        for layer in range(model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (
            model_attn_modules[layer],
            get_direction_ablation_output_hook(
                direction=direction,
                ablation_intensity=ablation_intensity,
                batch_indices=batch_indices,
            ),
        )
        for layer in range(model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (
            model_mlp_modules[layer],
            get_direction_ablation_output_hook(
                direction=direction,
                ablation_intensity=ablation_intensity,
                batch_indices=batch_indices,
            ),
        )
        for layer in range(model.config.num_hidden_layers)
    ]

    return fwd_pre_hooks, fwd_hooks
