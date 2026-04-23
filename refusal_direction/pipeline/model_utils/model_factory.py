import torch

from pipeline.model_utils.model_base import ModelBase


class StandardModel(ModelBase):
    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )


def construct_model_base(model_name_or_path: str) -> ModelBase:
    assert any(
        [
            family in model_name_or_path.lower()
            for family in [
                "llama-3",
                "qwen2.5",
                "qwen3",
                "gemma-2",
                "phi-4",
                "mistral",
                "gpt-oss",
                "olmo-3",
            ]
        ]
    )
    return StandardModel(model_name_or_path)
