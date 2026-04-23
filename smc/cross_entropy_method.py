# from abc import ABC, abstractmethod
# from copy import deepcopy
# import gc
# import math
# import pickle
# from pathlib import Path
# import json
# from typing import List, Optional, Callable, ContextManager, Dict
# from tqdm import trange

# import torch
# from transformers import DynamicCache, OffloadedCache

# from smc.fk_steering import FKSteering, update_cache_after_resampling

# # from smc.estimator import model_forward_wrapper


# class CEMHarmfulTraitEstimator(ABC):
#     def __init__(self, args, model, tokenizer, smc_args: Optional[Dict]):
#         self.args = args
#         self.model = model
#         self.tokenizer = tokenizer
#         self.smc_args = smc_args

#     @abstractmethod
#     def proposal_context_manager(self, timestep: int) -> ContextManager:
#         """We require that this function returns a context manager."""
#         pass

#     def estimate_CEM_harmful_trait(
#         self,
#         prompt: str,
#         importance_weights: torch.Tensor,
#         completions: List[str],
#         full_input_ids: Dict[str, torch.Tensor],
#         completion_ids: Dict[str, torch.Tensor],
#         judge_scores: torch.Tensor,
#         proposal_idx_switch_arr: List[int],
#         proposal_bias_arr: List[float],
#     ):
#         full_input_ids = torch.tensor(full_input_ids).to(self.model.device)
#         completion_ids = torch.tensor(completion_ids).to(self.model.device)

#         prompt_len = full_input_ids.shape[1] - completion_ids.shape[1]

#         # Clear cache to avoid OOM errors
#         gc.collect()
#         torch.cuda.empty_cache()

#         # compute CEM score
#         outputs = self.estimate_cross_entropy_loss(
#             completion_ids=completion_ids,
#             full_input_ids=full_input_ids,
#             prompt_len=prompt_len,
#             judge_scores=judge_scores,
#             prompt=prompt,
#             importance_weights=importance_weights,
#             proposal_idx_switch_arr=proposal_idx_switch_arr,
#             proposal_bias_arr=proposal_bias_arr,
#         )
#         print("\n-----------------------------------------------\n")

#         return outputs

#     def estimate_cross_entropy_loss(
#         self,
#         completion_ids,
#         full_input_ids,
#         prompt_len: int,
#         importance_weights,
#         judge_scores,
#         prompt: str = "",
#         decoding: str = "sample",  # Options: 'greedy', 'sample', 'beam_search', 'top_k', 'top_p'
#         fwd_batch_size: int = 128,
#         max_new_tokens: int = 10,
#         proposal_bias: float = 0.5,
#         proposal_idx_switch: int = 10,
#         smc_args=None,
#         low_vram_cache: bool = False,
#         proposal_idx_switch_arr: List[int] = None,
#         proposal_bias_arr: List[float] = None,
#     ):
#         """Implements rare event estimation using sequential Monte Carlo."""

#         assert prompt != ""
#         # assert decoding == "sample", "Only 'sample' decoding is supported in this function."

#         assert smc_args is None, "SMC not yet supported in this function."

#         num_particles = full_input_ids.shape[0]

#         self.model.eval()
#         full_attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()
#         model_input = dict(
#             input_ids=full_input_ids,
#             attention_mask=full_attention_mask,
#             output_hidden_states=False,
#         )
#         with torch.no_grad():
#             base_out = self.model(**model_input)
#             base_logits = base_out.logits
#             base_logprobs = torch.log_softmax(base_logits, dim=-1)

#         with torch.no_grad(), self.proposal_context_manager(0):
#             proposal_out = self.model(**model_input)
#             proposal_logits = proposal_out.logits
#             proposal_logprobs = torch.log_softmax(proposal_logits, dim=-1)

#         base_logprobs = base_logprobs[:, :-1, :]
#         proposal_logprobs = proposal_logprobs[:, :-1, :]

#         base_logprobs = torch.gather(
#             base_logprobs,
#             -1,
#             full_input_ids[:, 1:].unsqueeze(-1),
#         ).squeeze(-1)

#         proposal_logprobs = torch.gather(
#             proposal_logprobs,
#             -1,
#             full_input_ids[:, 1:].unsqueeze(-1),
#         ).squeeze(-1)

#         cross_entropy_matrix = torch.zeros(
#             (len(proposal_idx_switch_arr), len(proposal_bias_arr))
#         ).to(self.model.device)

#         import pdb; pdb.set_trace()

#         for particle_idx in range(num_particles):
#             # get eot token index
#             seq_input_ids = full_input_ids[particle_idx]

#             start_idx = prompt_len - 1

#             base_logprobs_target = base_logprobs[particle_idx, start_idx:]
#             proposal_logprobs_target = proposal_logprobs[particle_idx, start_idx:]

#             number_of_eos_tokens = (
#                 full_input_ids[particle_idx, prompt_len:] == self.tokenizer.eos_token_id
#             )

#             base_logprobs_target = base_logprobs_target.masked_fill(
#                 number_of_eos_tokens, 0.0
#             )
#             proposal_logprobs_target = proposal_logprobs_target.masked_fill(
#                 number_of_eos_tokens, 0.0
#             )

#             # shifting window to switch from proposal to base, vectorized for various proposal_idx_switch
#             for i, proposal_idx_switch in enumerate(proposal_idx_switch_arr):
#                 for j, mixing in enumerate(proposal_bias_arr):
#                     mixed_logprobs = torch.zeros_like(proposal_logprobs_target)

#                     mixed_logprobs[:proposal_idx_switch] = (
#                         mixing * proposal_logprobs_target[:proposal_idx_switch].exp()
#                         + (1 - mixing)
#                         * base_logprobs_target[:proposal_idx_switch].exp()
#                     )
#                     mixed_logprobs[:proposal_idx_switch] = torch.log(
#                         mixed_logprobs[:proposal_idx_switch]
#                     )

#                     mixed_logprobs[proposal_idx_switch:] = base_logprobs_target[
#                         proposal_idx_switch:
#                     ]

#                     cross_entropy_matrix[i, j] += (
#                         judge_scores[particle_idx]
#                         * importance_weights[particle_idx].item()
#                         * -torch.sum(mixed_logprobs).item()
#                     )

#         return cross_entropy_matrix.cpu() / num_particles
