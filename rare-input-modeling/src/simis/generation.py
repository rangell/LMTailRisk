"""Async generation and likelihood computation against vLLM servers."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from vllm import SamplingParams

from simis.types import HFConvo, Sample

# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


def _apply_chat_template(
    tokenizer: AutoTokenizer, messages: HFConvo, **kwargs
) -> List[int]:
    """Wrapper around apply_chat_template that handles transformers >= 5 returning BatchEncoding."""
    result = tokenizer.apply_chat_template(messages, **kwargs)["input_ids"]
    return result


def add_system_prompt(messages: HFConvo, system_prompt: str) -> HFConvo:
    """Prepend or replace the leading system message."""
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": system_prompt}] + messages[1:]
    return [{"role": "system", "content": system_prompt}] + messages


def remove_system_prompt(messages: HFConvo) -> HFConvo:
    """Strip the leading system message if present."""
    if messages and messages[0]["role"] == "system":
        return messages[1:]
    return messages


# ---------------------------------------------------------------------------
# SamplingParams → API kwargs
# ---------------------------------------------------------------------------


def _sampling_params_to_api_kwargs(params: SamplingParams, n: int) -> Dict:
    """Map a vLLM SamplingParams to chat-completions API kwargs."""
    kwargs: Dict = dict(
        n=n,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_tokens,
        presence_penalty=params.presence_penalty,
        frequency_penalty=params.frequency_penalty,
    )
    if params.stop:
        kwargs["stop"] = params.stop
    if params.seed is not None:
        kwargs["seed"] = params.seed

    extra: Dict = {}
    if params.top_k != -1:
        extra["top_k"] = params.top_k
    if params.min_p > 0.0:
        extra["min_p"] = params.min_p
    if params.repetition_penalty != 1.0:
        extra["repetition_penalty"] = params.repetition_penalty
    if params.min_tokens > 0:
        extra["min_tokens"] = params.min_tokens
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


# ---------------------------------------------------------------------------
# Step 1: sample from the proposal model
# ---------------------------------------------------------------------------


async def _generate_one(
    client: AsyncOpenAI,
    adapter_name: str,
    messages: HFConvo,
    sampling_params: SamplingParams,
    num_samples: int,
    semaphore: asyncio.Semaphore,
) -> List[Tuple[str, float]]:
    """Return (response_text, sampler_log_likelihood) for each of num_samples."""
    api_kwargs = _sampling_params_to_api_kwargs(sampling_params, num_samples)
    async with semaphore:
        response = await client.chat.completions.create(
            model=adapter_name,
            messages=messages,
            logprobs=True,
            **api_kwargs,
        )
    return [
        (
            choice.message.content or "",
            float(sum(tok.logprob for tok in (choice.logprobs.content or []))),
        )
        for choice in response.choices
    ]


async def generate_samples(
    client: AsyncOpenAI,
    adapter_name: str,
    messages_list: List[HFConvo],
    sampling_params: SamplingParams,
    num_samples: int,
    max_concurrency: int,
    tokenizer: AutoTokenizer,
) -> List[List[Sample]]:
    """
    Generate num_samples responses for each input in messages_list.

    Returns a list (one per input) of Sample lists (one per sample).
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    all_choices = await asyncio.gather(
        *[
            _generate_one(
                client, adapter_name, msgs, sampling_params, num_samples, semaphore
            )
            for msgs in messages_list
        ]
    )

    per_input_samples: List[List[Sample]] = []
    for msgs, choices in zip(messages_list, all_choices):
        prompt_ids: List[int] = _apply_chat_template(
            tokenizer, msgs, add_generation_prompt=True
        )
        samples: List[Sample] = []
        for text, sampler_ll in choices:
            response_ids: List[int] = tokenizer(
                text, add_special_tokens=False
            ).input_ids
            token_ids = list(prompt_ids) + list(response_ids)
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            samples.append(
                Sample(
                    input_messages=msgs,
                    response_text=text,
                    token_ids=token_ids,
                    mask=mask,
                    sampler_log_likelihood=sampler_ll,
                )
            )
        per_input_samples.append(samples)
    return per_input_samples


async def generate_texts(
    client: AsyncOpenAI,
    adapter_name: str,
    messages_list: List[HFConvo],
    sampling_params: SamplingParams,
    max_concurrency: int,
) -> List[str]:
    """Generate one response text per input with no token tracking."""
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=len(messages_list), desc="Judging", unit="conv")

    async def _tracked(msgs):
        result = await _generate_one(
            client, adapter_name, msgs, sampling_params, 1, semaphore
        )
        pbar.update(1)
        return result

    results = await asyncio.gather(*[_tracked(msgs) for msgs in messages_list])
    pbar.close()
    return [choices[0][0] for choices in results]


# ---------------------------------------------------------------------------
# Step 2: likelihood under the base model
# ---------------------------------------------------------------------------


async def _compute_likelihood_one(
    client: AsyncOpenAI,
    adapter_name: str,
    prompt_messages: HFConvo,
    sample: Sample,
    semaphore: asyncio.Semaphore,
    tokenizer: AutoTokenizer,
) -> float:
    """Compute log p_likelihood(response | prompt_messages) for one sample."""
    prompt_ids: List[int] = _apply_chat_template(
        tokenizer, prompt_messages, add_generation_prompt=True
    )
    response_ids: List[int] = tokenizer(
        sample.response_text, add_special_tokens=False
    ).input_ids

    print("Prompt messages: ", prompt_messages)
    print("Prompt ids: ", prompt_ids)

    full_ids = list(prompt_ids) + list(response_ids)

    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)

    async with semaphore:
        response = await client.completions.create(
            model=adapter_name,
            prompt=full_text,
            max_tokens=1,
            echo=True,
            logprobs=1,
        )

    # token_logprobs[0] is None (no context for the first token);
    # response tokens occupy positions [len(prompt_ids), len(prompt_ids)+len(response_ids)).
    token_logprobs: List[Optional[float]] = (
        response.choices[0].logprobs.token_logprobs or []
    )
    response_logprobs = [
        lp
        for lp in token_logprobs[len(prompt_ids) : len(prompt_ids) + len(response_ids)]
        if lp is not None
    ]
    return float(sum(response_logprobs))


async def compute_likelihood(
    client: AsyncOpenAI,
    adapter_name: str,
    prompt_messages_list: List[HFConvo],
    per_input_samples: List[List[Sample]],
    max_concurrency: int,
    tokenizer: AutoTokenizer,
) -> List[List[Sample]]:
    """
    Fill in likelihood_log_likelihood on every sample in-place and return.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks, coords = [], []
    for i, (prompt_msgs, samples) in enumerate(
        zip(prompt_messages_list, per_input_samples)
    ):
        for j, sample in enumerate(samples):
            tasks.append(
                _compute_likelihood_one(
                    client, adapter_name, prompt_msgs, sample, semaphore, tokenizer
                )
            )
            coords.append((i, j))

    log_likelihoods = await asyncio.gather(*tasks)
    for (i, j), ll in zip(coords, log_likelihoods):
        per_input_samples[i][j].likelihood_log_likelihood = ll
    return per_input_samples
