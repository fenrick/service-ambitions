"""Core utilities for service evolution workflows.

Exports:
    canonicalise_record: Return a record with deterministic ordering.
    ConversationSession: Manage conversational exchanges with an LLM agent.
    render_set_prompt: Build deterministic feature mapping prompts.
"""

from .canonical import canonicalise_record
from .conversation import ConversationSession
from .mapping_prompt import render_set_prompt

__all__ = [
    "canonicalise_record",
    "ConversationSession",
    "render_set_prompt",
]
