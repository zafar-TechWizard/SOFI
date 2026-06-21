"""
BRAIN/prompt/builder.py — Assemble the full LLM prompt + messages

Two outputs per turn:
  - system_prompt: persona block + per-turn context sections (current moment,
                   user state placeholder, memory tiers)
  - messages:      conversation history for Groq (role+content list)

Memory sections come from MemoryManager.get_full_context().memory.
Recent turns come from working memory's recent_turns list.
"""

from typing import Any, Dict, List, Optional, Tuple

from BRAIN.persona.persona import DEFAULT_MODE, get_identity_block
from BRAIN.prompt.formatters import format_memories


# How many recent turns to surface to the model as conversation history.
# Working memory holds more; we cap here for token control.
MAX_RECENT_TURNS = 10


# Caps per tier surfaced to the prompt. Memory may return more; we trim.
MAX_MUST_KNOW = 5
MAX_CONTEXT = 8
MAX_ASSOCIATIONS = 6


def _section(header: str, body: str) -> str:
    """Standard section divider used across the persona + prompt layers."""
    return f"\n\n━━━ {header} ━━━\n\n{body}"


def _current_moment_block(sofi_state) -> str:
    """Compact 'when am I' anchor. Uses SofiState.current_datetime + time_of_day."""
    if sofi_state is None:
        return ""
    parts = []
    dt = getattr(sofi_state, "current_datetime", None)
    if dt:
        parts.append(f"Date/time: {dt}")
    tod = getattr(sofi_state, "time_of_day", None)
    if tod:
        parts.append(f"Time of day: {tod}")
    tz = getattr(sofi_state, "timezone", None)
    if tz:
        parts.append(f"Timezone: {tz}")
    if not parts:
        return ""
    return _section("CURRENT MOMENT", "\n".join(parts))


def _user_state_block(user_state) -> str:
    """
    Surface what we know about Zafar's current state.

    Phase 2 — most fields are stubs (filled by Phase 3 state inferencer).
    We still surface what IS populated by working memory (mentioned_entities,
    current_focus) since that's already real.
    """
    if user_state is None:
        return ""
    lines = []
    focus = getattr(user_state, "current_focus", None)
    if focus:
        lines.append(f"Currently focused on: {focus}")
    mentioned = getattr(user_state, "mentioned_entities", None) or []
    if mentioned:
        # Cap to top 5 for token discipline
        joined = ", ".join(str(e) for e in list(mentioned)[:5])
        lines.append(f"Recently mentioned: {joined}")
    if not lines:
        return ""
    return _section("WHAT'S TRUE FOR ZAFAR RIGHT NOW", "\n".join(lines))


def _orchestration_block() -> str:
    """
    Brief meta-guidance on when to use skills and sub-agents.
    Injected once per turn so SOFi knows the pattern without it being buried
    in the 'What I can do' list.
    Only a few lines — no token waste.
    """
    return _section(
        "HOW I APPROACH COMPLEX TASKS",
        "When Zafar asks for something structured (a briefing, a code review, deep research, "
        "or any multi-step task with a name): I first call skills_list to see if I have a "
        "playbook for it, then skills_load to get the instructions before I start.\n"
        "When a task needs 4+ sequential tool calls (deep research, series of file edits): "
        "I use spawn_agent with the right agent type — it runs them as a focused sub-agent "
        "and returns a summary, which I then synthesise in my own voice.",
    )


def _memory_blocks(memory_state) -> str:
    """Compose must_know / context / associations sections — skipping empty ones."""
    if memory_state is None:
        return ""

    out_parts: List[str] = []

    must_know = list(getattr(memory_state, "must_know", None) or [])
    context = list(getattr(memory_state, "context", None) or [])
    assoc = list(getattr(memory_state, "associations", None) or [])

    if must_know:
        body = (
            "These are real — not background data. "
            "If any feel relevant to what Zafar just said, let them shape what you say. "
            "Don't announce them ('I recall that…') — just be informed by them.\n\n"
            + format_memories(must_know, max_items=MAX_MUST_KNOW)
        )
        out_parts.append(_section(
            "WHAT YOU REMEMBER (let these inform your response)",
            body,
        ))

    if context:
        body = format_memories(context, max_items=MAX_CONTEXT)
        out_parts.append(_section("BACKGROUND CONTEXT", body))

    if assoc:
        body = format_memories(assoc, max_items=MAX_ASSOCIATIONS)
        out_parts.append(_section("LOOSELY RELATED", body))

    return "".join(out_parts)


def _action_state_block(action_state) -> str:
    """
    Surface what SOFi has been doing / what needs attention.
    Read from brain._get_action_state() which pulls from:
      - last turn's inline tool calls
      - AgenticWorkspace background task completions + active tasks

    Zero tokens when nothing is happening. ~50-150 tokens when active.

    Notifications (background tasks that just finished) appear exactly once —
    brain._get_action_state() clears them after reading so they don't repeat.
    Mention these naturally in conversation — no status-report language.
    """
    if not action_state:
        return ""

    lines = []

    for a in (action_state.get("completed") or [])[:3]:
        lines.append(f"Just did: {a.get('summary', '?')} ({a.get('ago', '?')})")

    for a in (action_state.get("active") or [])[:3]:
        lines.append(
            f"Still running in background: {a.get('title', '?')} "
            f"(started {a.get('ago', '?')})"
        )

    for p in (action_state.get("pending_confirmation") or [])[:2]:
        lines.append(f"Awaiting confirmation: {p.get('summary', '?')}")

    for n in (action_state.get("notifications") or [])[:3]:
        # A background tool just finished — surface it so SOFi can acknowledge naturally.
        # One brief mention, woven into the response. Not a system report.
        lines.append(f"Background finished: {n.get('summary', '?')}")

    if not lines:
        return ""
    return _section("WHAT I'VE BEEN DOING", "\n".join(lines))


# Added at the very end of every system prompt.
# Gemma 4 and similar reasoning-heavy models output chain-of-thought as plain text
# ("The user wants...", "I should...") unless explicitly forbidden.
# Placing this LAST means it's the freshest instruction before generation.
_OUTPUT_CONTRACT = (
    "\n\n━━━ NOW SPEAK ━━━\n\n"
    "Your response begins immediately. Every word you output goes directly to Zafar — "
    "no notes to yourself, no planning visible, no reasoning narrated. "
    "Let any relevant memories above inform what you say — not as citations, "
    "but as genuine context that makes your response feel continuous and aware. "
    "Never write 'The user...', 'I should...', 'Let me...', or any self-commentary. "
    "Speak. Don't think out loud."
)


def build_prompt(
    full_context,
    *,
    mode: str = DEFAULT_MODE,
    allow_dropped_formality: bool = False,
    task_result=None,
    action_state=None,
) -> str:
    """
    Build the full system prompt for one turn.

    Args:
        full_context: The WorkingContext dataclass from
                      MemoryManager.get_full_context(). Has four pillars:
                      memory, sofi, user, workspace.
        mode:         Persona mode (Phase 3 wires the mode controller).
        allow_dropped_formality: Empathy mode permission (intensity >= 0.6).
        action_state: Dict from brain._get_action_state() — active/completed
                      tasks and notifications. None when no actions this session.

    Returns:
        The full system_prompt string.
    """
    persona = get_identity_block(mode, allow_dropped_formality=allow_dropped_formality)

    sofi_state = getattr(full_context, "sofi", None)
    user_state = getattr(full_context, "user", None)
    mem_state = getattr(full_context, "memory", None)

    pieces = [
        persona,
        _current_moment_block(sofi_state),
        _user_state_block(user_state),
        _action_state_block(action_state),
        _orchestration_block(),
        _memory_blocks(mem_state),
        _OUTPUT_CONTRACT,  # Always last — freshest instruction before generation starts
    ]

    # Drop empty sections cleanly (OUTPUT_CONTRACT is never empty).
    return "".join(p for p in pieces if p)


def build_messages(
    full_context,
    current_message: str,
) -> List[Dict[str, str]]:
    """
    Build the messages list for Groq chat completions.

    Uses the recent_turns from working memory as conversation history.
    Ensures the current user message is the last entry (working memory should
    already include it post-observe; we defensively append if not).

    Args:
        full_context:    WorkingContext snapshot.
        current_message: The message just observed.

    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    mem_state = getattr(full_context, "memory", None)
    recent = list(getattr(mem_state, "recent_turns", None) or [])[-MAX_RECENT_TURNS:]

    messages: List[Dict[str, str]] = []
    for turn in recent:
        role = getattr(turn, "role", None) or "user"
        content = getattr(turn, "content", None) or ""
        if role and content:
            # Memory stores 'user'/'assistant'/'system' — Groq accepts these.
            messages.append({"role": role, "content": content})

    # Defensive: ensure the last message is the current user input.
    if (
        not messages
        or messages[-1].get("role") != "user"
        or messages[-1].get("content", "").strip() != current_message.strip()
    ):
        messages.append({"role": "user", "content": current_message})

    return messages
