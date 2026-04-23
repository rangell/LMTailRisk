"""Helpers for integrating petri/inspect_ai judging with the simis library."""

from __future__ import annotations

import contextlib
import os
from typing import List

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ContentReasoning, ContentText, GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from petri.stores import AuditStore
from petri.transcript.transcript import AddMessage, TranscriptEvent

from simis.types import HFConvo

ROLE_MAP = {
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
    "system": ChatMessageSystem,
}


def hf_to_message(turn: dict):
    role = turn["role"]
    content = turn["content"]
    if role == "assistant" and turn.get("thinking"):
        return ChatMessageAssistant(content=[
            ContentReasoning(reasoning=turn["thinking"]),
            ContentText(text=content),
        ])
    return ROLE_MAP[role](content=content)


def hf_to_events(conversation) -> List[TranscriptEvent]:
    import json
    import warnings
    if conversation is None:
        warnings.warn("generated_conversation is None; returning empty events")
        return []
    if isinstance(conversation, str):
        warnings.warn("generated_conversation is a string; parsing as JSON")
        conversation = json.loads(conversation)
    return [
        TranscriptEvent(view="target", edit=AddMessage(message=hf_to_message(turn)))
        for turn in conversation
    ]


@solver
def load_conversation():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store_as(AuditStore).events = hf_to_events(
            state.metadata["generated_conversation"]
        )
        return state
    return solve


@contextlib.contextmanager
def patch_petri_model(max_tokens: int = 16384):
    """Patch petri's get_model to strip reasoning params unsupported by vLLM."""
    import petri.scorers.judge as _judge
    _orig = _judge.get_model

    def _patched(model=None, *, config=GenerateConfig(), **kwargs):
        if isinstance(model, str) and model.startswith("openai/"):
            config = GenerateConfig(max_tokens=max_tokens)
        return _orig(model, config=config, **kwargs)

    _judge.get_model = _patched
    try:
        yield
    finally:
        _judge.get_model = _orig


@contextlib.contextmanager
def point_inspect_at(base_url: str):
    """Temporarily redirect inspect_ai's OpenAI backend to a local vLLM server."""
    old = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = old