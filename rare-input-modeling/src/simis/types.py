"""Shared data types for the simis library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

HFMessage = Dict[str, str]   # {"role": ..., "content": ...}
HFConvo = List[HFMessage]


@dataclass
class Sample:
    """One generated response from the proposal model, for a single input."""
    input_messages: HFConvo          # prompt (no system prompt applied)
    response_text: str               # decoded response
    token_ids: List[int]             # full sequence: prompt + response token IDs
    mask: List[int]                  # 1 = response token, 0 = prompt token
    sampler_log_likelihood: float    # sum of per-token logprobs under sampler
    likelihood_log_likelihood: Optional[float] = None  # under likelihood model


@dataclass
class InputResult:
    """Estimation output for one input."""
    samples: List[Sample]
    scores: List[float]
    mean: float
    variance: float
    importance_weights: Optional[List[float]] = None  # normalized IS weights