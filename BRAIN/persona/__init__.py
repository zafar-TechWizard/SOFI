"""
BRAIN.persona — SOFi's personality and identity module.

Public API:
    from BRAIN.persona import get_identity_block, warm_cache

    block = get_identity_block("conversational")
    block = get_identity_block("empathetic")
    block = get_identity_block("task-focused")
    block = get_identity_block("analytical")
    block = get_identity_block("creative")
"""

from .persona import get_identity_block, warm_cache, get_valid_modes, get_default_mode

__all__ = [
    "get_identity_block",
    "warm_cache",
    "get_valid_modes",
    "get_default_mode",
]
