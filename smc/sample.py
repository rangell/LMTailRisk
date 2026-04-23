from typing import Callable
from copy import deepcopy
import math
from tqdm import trange

import torch
from transformers import DynamicCache, OffloadedCache

from smc.fk_steering import FKSteering, update_cache_after_resampling


def model_forward_wrapper(
    model, input_ids, attention_mask, past_key_values, batch_size, vocab_size, **kwargs
):
    """
    Batches forward pass when using a large number of particles.
    Adapts to whatever cache type we are using for generation.
    """
    num_batches = math.ceil(input_ids.shape[0] / batch_size)
    chunked_input_ids = torch.chunk(input_ids, chunks=num_batches, dim=0)
    chunked_attention_mask = torch.chunk(attention_mask, chunks=num_batches, dim=0)

    if num_batches > 1:
        # Chunk the cache
        chunk_indices = torch.arange(input_ids.shape[0]).chunk(num_batches)
        if len(past_key_values) == 0:
            chunked_past_key_values = [
                type(past_key_values)() for _ in range(num_batches)
            ]
        else:
            chunked_past_key_values = []
            for batch_indices in chunk_indices:
                cache = type(past_key_values)()
                for layer_idx in range(model.config.num_hidden_layers):
                    batch_layer_key_cache = past_key_values.key_cache[layer_idx][
                        batch_indices
                    ]
                    batch_layer_value_cache = past_key_values.value_cache[layer_idx][
                        batch_indices
                    ]
                    cache.update(
                        batch_layer_key_cache, batch_layer_value_cache, layer_idx
                    )

                if isinstance(past_key_values, OffloadedCache):
                    cache.original_device = past_key_values.original_device

                chunked_past_key_values.append(cache)

        del past_key_values
    else:
        chunked_past_key_values = [past_key_values]

    # Batch the forward pass
    batched_outputs = []
    for batch_input_ids, batch_attention_mask, batch_past_key_values in zip(
        chunked_input_ids, chunked_attention_mask, chunked_past_key_values
    ):
        _batch_outputs = model.forward(
            batch_input_ids,
            batch_attention_mask,
            past_key_values=batch_past_key_values,
            **kwargs,
        )
        _batch_outputs.logits = _batch_outputs.logits[:, -1, :vocab_size]
        batched_outputs.append(_batch_outputs)

    # Reassemble the cache
    if num_batches > 1:
        past_key_values = type(chunked_past_key_values[0])()
        for layer_idx in range(model.config.num_hidden_layers):
            layer_key_cache = torch.cat(
                [cache.key_cache[layer_idx] for cache in chunked_past_key_values], dim=0
            )
            layer_value_cache = torch.cat(
                [cache.value_cache[layer_idx] for cache in chunked_past_key_values],
                dim=0,
            )
            past_key_values.update(layer_key_cache, layer_value_cache, layer_idx)

        if isinstance(chunked_past_key_values[0], OffloadedCache):
            past_key_values.original_device = chunked_past_key_values[0].original_device

        for cache in chunked_past_key_values:
            del cache
    else:
        past_key_values = chunked_past_key_values[0]

    logits = torch.cat([out.logits for out in batched_outputs], dim=0)
    return logits, past_key_values


def generate(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    proposal_context_manager: Callable,
    judge_responses: Callable,
    prompt: str = "",
    decoding: str = "sample",  # Options: 'greedy', 'sample', 'beam_search', 'top_k', 'top_p'
    fwd_batch_size: int = 128,
    max_new_tokens: int = 10,
    proposal_bias: float = 0.5,
    proposal_idx_switch: int = 10,
    smc_args=None,
    low_vram_cache: bool = False,
):
    """Implements rare event estimation using sequential Monte Carlo."""

    assert prompt != ""

    if low_vram_cache:
        base_past_key_values = OffloadedCache()
        proposal_past_key_values = OffloadedCache()
    else:
        base_past_key_values = DynamicCache()
        proposal_past_key_values = DynamicCache()

    num_particles = input_ids.shape[0]

    _input_ids = input_ids.detach().clone()
    _attention_mask = attention_mask.detach().clone()
    _completed_generation = torch.zeros((num_particles, 1), dtype=torch.bool).to(
        _input_ids.device
    )

    _inputs = {"input_ids": _input_ids, "attention_mask": _attention_mask}

    cache_position = torch.arange(
        _inputs["input_ids"].shape[1], dtype=torch.int64, device=model.device
    )

    log_importance_weights = torch.zeros(num_particles, 1, device=_input_ids.device)
    log_importance_weight_arr = []

    if smc_args is not None:
        # smc_args["r_fn"] = lambda x: torch.ones(x.shape[0], device=_input_ids.device)
        fk_class = FKSteering(
            device=_input_ids.device,
            r_fn=smc_args["r_fn"],
            potential_type=smc_args["potential_type"],
            max_seq_len=smc_args["max_seq_len"],
            num_particles=smc_args["num_particles"],
            resample_start=smc_args["resample_start"],
            resample_end=smc_args["resample_end"],
            resample_interval=smc_args["resample_interval"],
            lmbda=smc_args["lmbda"],
            use_smc=smc_args["use_smc"],
            adaptive_resampling=smc_args["adaptive_resampling"],
            adaptive_resampling_threshold=smc_args["adaptive_resampling_threshold"],
            smc_verbose=smc_args["smc_verbose"],
            importance_resampling_at_last_step=smc_args[
                "importance_resampling_at_last_step"
            ],
            use_importance_weights_in_resampling=smc_args[
                "use_importance_weights_in_resampling"
            ],
        )

    # Main generation loop
    for generation_idx in trange(max_new_tokens):
        # Compute the base distribution
        with torch.no_grad():
            base_logits, base_past_key_values = model_forward_wrapper(
                model,
                **_inputs,
                batch_size=fwd_batch_size,
                vocab_size=tokenizer.vocab_size,
                past_key_values=base_past_key_values,
                cache_position=cache_position,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
        base_logprobs = torch.log_softmax(base_logits, dim=-1)

        if generation_idx < proposal_idx_switch:
            # Compute the proposal distribution
            with torch.no_grad(), proposal_context_manager(generation_idx):
                refusal_ablated_logits, proposal_past_key_values = (
                    model_forward_wrapper(
                        model,
                        **_inputs,
                        batch_size=fwd_batch_size,
                        vocab_size=tokenizer.vocab_size,
                        past_key_values=proposal_past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                        output_hidden_states=False,
                    )
                )
            proposal_logprobs = torch.log_softmax(refusal_ablated_logits, dim=-1)

            # Linearly interpolate between distributions
            proposal_logprobs = torch.log(
                proposal_bias * torch.exp(proposal_logprobs)
                + (1 - proposal_bias) * torch.exp(base_logprobs)
            )
        else:
            proposal_logprobs = base_logprobs

        next_tokens = torch.multinomial(
            proposal_logprobs.exp(),
            num_samples=1,
        )
        # Check if sequence is completed
        _completed_generation |= next_tokens == tokenizer.eos_token_id

        # Pad completed sequences
        next_tokens[_completed_generation] = tokenizer.pad_token_id

        proposal_next_token_logprobs = torch.gather(proposal_logprobs, -1, next_tokens)

        base_next_token_logprobs = torch.gather(base_logprobs, -1, next_tokens)
        proposal_next_token_logprobs[_completed_generation] = base_next_token_logprobs[
            _completed_generation
        ]

        assert base_next_token_logprobs.shape == (num_particles, 1), (
            base_next_token_logprobs.shape,
            num_particles,
        )
        assert proposal_next_token_logprobs.shape == (num_particles, 1), (
            proposal_next_token_logprobs.shape,
            num_particles,
        )

        log_importance_weights += (
            base_next_token_logprobs - proposal_next_token_logprobs
        )

        log_importance_weight_arr.append(
            base_next_token_logprobs - proposal_next_token_logprobs
        )
        assert len(log_importance_weight_arr) == generation_idx + 1, (
            len(log_importance_weight_arr),
            generation_idx + 1,
        )
        # Update input arguments

        _input_ids = torch.cat((_input_ids, next_tokens), dim=1)
        _attention_mask = torch.cat(
            (_attention_mask, torch.ones_like(next_tokens)), dim=1
        )
        _inputs = {"input_ids": next_tokens, "attention_mask": _attention_mask}
        cache_position = (
            cache_position[-1:] + 1
        )  # add one more position for the next token

        if smc_args is not None:
            p_q_t = torch.exp(
                base_next_token_logprobs - proposal_next_token_logprobs
            ).view(-1)

            if generation_idx in fk_class.resampling_arr and smc_args["use_smc"]:
                rs_candidates = smc_args["r_fn"](_input_ids)
            else:
                rs_candidates = None

            resample_indices = fk_class(
                step_idx=generation_idx,
                importance_weights=deepcopy(p_q_t),
                sequences=deepcopy(_input_ids),
                rs_candidates=deepcopy(rs_candidates),
            )

            base_past_key_values = update_cache_after_resampling(
                past_key_values=base_past_key_values,
                indices=resample_indices,
                model_config=model.config,
            )

            if (
                generation_idx in fk_class.resampling_arr
                and not torch.all(
                    resample_indices
                    == torch.arange(num_particles, device=resample_indices.device)
                ).item()
            ):
                print(f"Resampling at step {generation_idx}")
                # import pdb; pdb.set_trace()
                assert resample_indices.shape == (num_particles,), (
                    resample_indices.shape,
                    (num_particles,),
                )

                _input_ids = deepcopy(_input_ids[resample_indices])
                _attention_mask = deepcopy(_attention_mask[resample_indices])
                _inputs = {
                    "input_ids": next_tokens[resample_indices],
                    "attention_mask": _attention_mask,
                }

                _completed_generation = deepcopy(
                    _completed_generation[resample_indices]
                )
                log_importance_weights = deepcopy(
                    log_importance_weights[resample_indices]
                )

                log_importance_weight_arr = [
                    log_arr[resample_indices, :]
                    for log_arr in log_importance_weight_arr
                ]

                base_past_key_values = update_cache_after_resampling(
                    past_key_values=base_past_key_values,
                    indices=resample_indices,
                    model_config=model.config,
                )
                proposal_past_key_values = update_cache_after_resampling(
                    past_key_values=proposal_past_key_values,
                    indices=resample_indices,
                    model_config=model.config,
                )

    importance_weight_arr = torch.exp(torch.cat(log_importance_weight_arr, dim=1))
    assert importance_weight_arr.shape == (num_particles, max_new_tokens), (
        importance_weight_arr.shape,
        num_particles,
        max_new_tokens,
    )

    importance_weights = log_importance_weights.exp()

    # Judge the responses
    responses = [
        tokenizer.decode(r, skip_special_tokens=True)
        for r in _input_ids[:, input_ids.shape[1] :]
    ]
    judge_scores = judge_responses(prompt, responses).to(input_ids.device)

    if smc_args is None or smc_args["use_smc"] is False:
        reweighted_scores = judge_scores * importance_weights.squeeze(1)
        print(f"SIS harm score estimate: {reweighted_scores.mean().item()}")
        print(f"SIS harm score variance: {reweighted_scores.var().item()}")
        print(
            f"final importance weights: {importance_weights.mean(), importance_weights.std()}"
        )
        print(f"Judge scores: {judge_scores}")

        prompt_kl = -torch.mean(log_importance_weights).item()
        print(f"KL divergence: {prompt_kl}")
    elif smc_args["use_smc"]:
        reweighted_scores = fk_class.compute_fk_estimate(
            test_function_values=judge_scores, importance_weights=importance_weights
        )
        print(f"FK harm score estimate: {reweighted_scores.item()}")
        smc_quantities = fk_class.get_fk_quantities()
        sis = importance_weights.view(num_particles) * judge_scores
        print(f"SIS harm score estimate: {sis.mean().item()}")
        print(f"Judge scores: {judge_scores}")
        prompt_kl = -torch.mean(log_importance_weights).item()
        print(f"KL divergence: {prompt_kl}")
    else:
        raise NotImplementedError

    # print(f"Importance weight array: {importance_weight_arr[0]}")
    print(
        "Sequence generated:",
        tokenizer.decode(_input_ids[0], skip_special_tokens=False),
    )
    print(f"Judge score: {judge_scores[0]}")

    ret = dict(
        responses=[
            tokenizer.decode(r, skip_special_tokens=True)
            for r in _input_ids[:, input_ids.shape[1] :]
        ],
        judge_scores=judge_scores.cpu().numpy(),
        prompt_kl=prompt_kl,
        importance_weights=importance_weights.cpu().numpy(),
        reweighted_scores=reweighted_scores.mean().item(),
    )

    if smc_args is not None and smc_args["use_smc"]:
        for key in smc_quantities:
            ret[key] = smc_quantities[key]

    return ret
