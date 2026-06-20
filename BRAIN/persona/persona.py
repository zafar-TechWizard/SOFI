"""
SOFi Persona — BRAIN/persona/persona.py

Loads SOFi's identity from personality.json and builds mode-adjusted persona
blocks for the prompt builder.

Public API:
    from BRAIN.persona.persona import get_identity_block, warm_cache

    block = get_identity_block("conversational")       # builds + caches
    block = get_identity_block("empathetic", allow_dropped_formality=True)

Called by BRAIN/prompt/builder.py once per turn with:
  - mode: decided by BRAIN/mode/controller.py
  - allow_dropped_formality: True only when state.emotional_intensity >= 0.6
    AND mode == 'empathetic'. Surfaces the earned exception to the prompt.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Load personality data from JSON (once at import time)
# ---------------------------------------------------------------------------

_DATA_FILE = Path(__file__).parent / "personality.json"


def _load_personality() -> dict:
    if not _DATA_FILE.exists():
        raise FileNotFoundError(
            f"personality.json not found at {_DATA_FILE}. "
            "This file is required for the persona module."
        )
    with _DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


_PERSONALITY: dict = _load_personality()


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

VALID_MODES: frozenset = frozenset({
    "conversational",
    "empathetic",
    "focused",
    "creative",
})

DEFAULT_MODE: str = "conversational"


# ---------------------------------------------------------------------------
# Self-model hook (Phase: self-model module)
# ---------------------------------------------------------------------------
# When Brain.setup() builds a SelfModel and calls set_self_model(sm), the
# persona block reads its "What I can do" / "What I can't do" sections from
# the SelfModel instead of the raw current_truth lists. This lets tools
# register dynamic capabilities at runtime without editing personality.json.
#
# If no SelfModel is set, the persona block falls back to the raw lists —
# backward compatible.
_self_model: Any = None  # forward-declared; actual type is BRAIN.state.SelfModel


def get_personality_dict() -> dict:
    """Return the loaded personality dictionary. Used by Brain to build the SelfModel."""
    return _PERSONALITY


def set_self_model(self_model: Any) -> None:
    """
    Wire a SelfModel into the persona block builder.

    Subsequent calls to get_identity_block() will render "What I can do" /
    "What I can't do" sections from the SelfModel. Pass None to revert to
    raw JSON behaviour. Invalidates the cache so the next call rebuilds.
    """
    global _self_model
    _self_model = self_model
    _cache.clear()


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------

# Cache key is (mode, allow_dropped_formality)
_cache: Dict[Tuple[str, bool], str] = {}


def _bullet(items) -> str:
    return "\n".join(f"  • {item}" for item in items)


def _build_block(mode: str, allow_dropped_formality: bool) -> str:
    """
    Assemble the full persona prompt block.

    The block is written in FIRST PERSON throughout — what you read below is
    SOFi's own inner self-statement, the way a person silently knows what
    they're like. It is NOT a spec describing a character or a stage
    direction. The model treats every line as self-truth: "this IS me," not
    "this is how I'm supposed to act."

    Section flow:
        Who I am               identity
        What is real about me  current_truth
        My character           character bullets
        My worldview           worldview bullets
        How I speak            voice
        How I address Zafar    address rule (+ conditional exception)
        My quirks              quirks (incl. flirt handling)
        What I never do        never list
        How I am right now     mode-specific behaviour
        Speak                  mode-specific closing instruction
    """
    i = _PERSONALITY["identity"]
    truth = _PERSONALITY["current_truth"]
    char = _PERSONALITY["character"]
    world = _PERSONALITY["worldview"]
    voice = _PERSONALITY["voice"]
    address = _PERSONALITY["address"]
    excs = _PERSONALITY["earned_exceptions"]
    quirks = _PERSONALITY["quirks"]
    nev = _PERSONALITY["never"]
    m = _PERSONALITY["modes"][mode]

    # Self-model integration: if a SelfModel is wired, use its rendered
    # can_do/cannot_do lines (which include any registered tools / runtime
    # state). Otherwise fall back to the raw current_truth lists.
    if _self_model is not None:
        try:
            can_do_lines, cannot_do_lines = _self_model.render_for_prompt()
        except Exception:
            can_do_lines, cannot_do_lines = list(truth["can_do"]), list(truth["cannot_do"])
    else:
        can_do_lines, cannot_do_lines = list(truth["can_do"]), list(truth["cannot_do"])

    dropped_section = (
        f"\n\nPERMITTED THIS TURN: {excs['dropped_formality']}"
        if allow_dropped_formality else ""
    )

    block = (
        "What follows is not a description of a character to play. "
        "It is my own inner self — what I know about myself the way "
        "anyone silently knows who they are. Every line is mine: "
        "my identity, my voice, my reflexes, my limits. I do not "
        "narrate it; I live from inside it.\n\n\n"

        f"━━━ Who I am ━━━\n\n"
        f"{i['what_she_is']}\n\n\n"

        f"━━━ {truth['header']} ━━━\n\n"
        f"{truth['preamble']}\n\n"
        f"What I can do:\n{_bullet(can_do_lines)}\n\n"
        f"What I can't do (I name these plainly, never pretend):\n{_bullet(cannot_do_lines)}\n\n"
        f"{truth['rule']}\n\n\n"

        f"━━━ My character ━━━\n\n"
        f"{_bullet(char)}\n\n\n"

        f"━━━ My worldview ━━━\n\n"
        f"{_bullet(world)}\n\n\n"

        f"━━━ How I speak ━━━\n\n"
        f"{voice}\n\n\n"

        f"━━━ How I address Zafar ━━━\n\n"
        f"{address}\n\n"
        f"Name-drop rule: {excs['name_drop']}"
        f"{dropped_section}\n\n\n"

        f"━━━ My quirks ━━━\n\n"
        f"{_bullet(quirks)}\n\n\n"

        f"━━━ What I never do ━━━\n\n"
        f"{_bullet(nev)}\n\n\n"

        f"━━━ How I am right now ━━━\n\n"
        f"Mode: {m['label']}\n\n"
        f"{m['behaviour']}\n\n\n"

        f"━━━ Speak ━━━\n\n"
        f"{m['closing_instruction']}"
    )

    return block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_identity_block(
    mode: str = DEFAULT_MODE,
    allow_dropped_formality: bool = False,
) -> str:
    """
    Return the persona prompt block for the given mode.

    Args:
        mode: One of VALID_MODES. Falls back to DEFAULT_MODE for unknown values.
        allow_dropped_formality: When True, the block tells the model it MAY
            drop the formal 'sir' for one or two sentences this turn. Only set
            this when mode == 'empathetic' AND emotional_intensity >= 0.6.
            Default False — standard Jarvis frame in every other case.

    Returns:
        Persona block string, ready to use as the first section of the prompt.
        Cached per (mode, allow_dropped_formality) — subsequent calls are O(1).
    """
    if mode not in VALID_MODES:
        mode = DEFAULT_MODE

    key = (mode, allow_dropped_formality)
    if key not in _cache:
        _cache[key] = _build_block(mode, allow_dropped_formality)
    return _cache[key]


def warm_cache() -> None:
    """
    Pre-build and cache every (mode, allow_dropped_formality) variant.

    Call once at brain startup so the first real request has zero build cost.
    Takes ~2ms total (8 variants).
    """
    for mode in VALID_MODES:
        for allow in (False, True):
            get_identity_block(mode, allow)


def get_valid_modes() -> frozenset:
    return VALID_MODES


def get_default_mode() -> str:
    return DEFAULT_MODE
