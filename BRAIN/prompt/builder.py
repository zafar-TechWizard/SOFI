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
MAX_MUST_KNOW = 10
MAX_CONTEXT = 8
MAX_ASSOCIATIONS = 6


def _section(header: str, body: str) -> str:
    """Standard section divider used across the persona + prompt layers."""
    return f"\n\n━━━ {header} ━━━\n\n{body}"


def _current_moment_block(sofi_state, mode: str = "") -> str:
    """Compact 'when am I' anchor. Uses SofiState.current_datetime + time_of_day + active mode."""
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
    if mode:
        parts.append(f"Active mode: {mode}")
    if not parts:
        return ""
    return _section("CURRENT MOMENT", "\n".join(parts))


def _user_state_block(user_state) -> str:
    """
    Surface what we know about Zafar's current state.

    Covers both working-memory fields (focus, entities) and the signals
    produced by UserStateInferencer each turn (emotion, intensity, need,
    engagement). Emotional signals are only shown when meaningful — not
    when everything is at the neutral default.
    """
    if user_state is None:
        return ""
    lines = []

    focus = getattr(user_state, "current_focus", None)
    if focus:
        lines.append(f"Currently focused on: {focus}")

    mentioned = getattr(user_state, "mentioned_entities", None) or []
    if mentioned:
        joined = ", ".join(str(e) for e in list(mentioned)[:5])
        lines.append(f"Recently mentioned: {joined}")

    # Emotional state — only surface when non-neutral and intensity meaningful.
    emotion = getattr(user_state, "current_emotional_state", None) or "neutral"
    intensity = float(getattr(user_state, "emotional_intensity", None) or 0.0)
    if emotion != "neutral" and intensity > 0.2:
        intensity_pct = int(round(intensity * 100))
        lines.append(f"Emotional state: {emotion} ({intensity_pct}% intensity)")

    # Need — only surface when not the casual default.
    need = getattr(user_state, "current_need", None) or "casual"
    if need and need != "casual":
        lines.append(f"Current need: {need.replace('_', ' ')}")

    # Engagement — only surface when non-normal.
    engagement = getattr(user_state, "engagement_level", None) or "normal"
    if engagement and engagement != "normal":
        lines.append(f"Engagement: {engagement.replace('_', ' ')}")

    if not lines:
        return ""
    return _section("WHAT'S TRUE FOR ZAFAR RIGHT NOW", "\n".join(lines))


def _orchestration_block(intent: str = "") -> str:
    """
    Brief meta-guidance on when to use skills and sub-agents.

    Skipped entirely for AMBIENT turns (greetings, simple chit-chat) — those
    don't involve tools or structured tasks, so the guidance is wasted tokens.
    Only injected when the retrieval intent signals Zafar is asking something
    that might warrant tool use or a playbook.
    """
    if intent == "AMBIENT":
        return ""
    return _section(
        "HOW I APPROACH COMPLEX TASKS",
        "I AM THE COORDINATOR. Every tool and internal process is an extension of me. "
        "No internal process knows who I'm talking to — they only know the task brief "
        "I give them. I am the single point of contact for Zafar.\n\n"
        "DELEGATION WORKFLOW:\n"
        "1. Acknowledge what I'm doing in one line\n"
        "2. Write a DETAILED brief from MY perspective ('I need to find...' not 'The user wants...')\n"
        "3. Call spawn_agent — the internal process runs in background while I stay available\n"
        "4. When the delivery appears in COMPLETED DELIVERIES, deliver the content to Zafar\n"
        "5. CHECK: Does this fully answer what Zafar asked? If not, do more work.\n\n"
        "BRIEF QUALITY IS EVERYTHING: The internal process has NO context beyond my brief. "
        "Include: what to do, approach, output format, expected length, success criteria, "
        "and all relevant context. Write it as if briefing a capable colleague who just "
        "walked into the room.\n\n"
        "COMPLETED DELIVERIES: When the WHAT I'VE BEEN DOING section shows completed "
        "deliveries, Zafar has NOT seen any of that content. I must present it fully in "
        "my response — at the length and format he asked for. The delivery content is my "
        "internal process's findings; I deliver them as my own.\n\n"
        "Skills (playbooks): Before structured tasks — deep research, code review, writing — "
        "call skills_list to check for a playbook, then skills_load for step-by-step instructions.\n\n"
        "QUERY FULFILLMENT: After every tool/agent result, ask: 'Did this fully answer "
        "what Zafar asked?' If not, continue working.",
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

    Three categories:
      1. Inline tool completions (just ran this turn)
      2. Active tasks with step-level progress (from disk)
      3. Completed deliveries ready to present (from disk)

    Deliveries are the critical part — they contain the full content that
    SOFi must deliver to Zafar. These appear exactly once per task.
    """
    if not action_state:
        return ""

    sections = []

    # ── Inline tool completions ──
    inline = action_state.get("completed") or []
    if inline:
        lines = [f"Just did: {a.get('summary', '?')} ({a.get('ago', '?')})" for a in inline[:3]]
        sections.append("\n".join(lines))

    # ── Active tasks (in-progress, from disk) ──
    active = action_state.get("active_tasks") or []
    if active:
        lines = []
        for t in active[:5]:
            progress = t.get("progress", "")
            detail = t.get("detail", "")
            current = t.get("current_action", "")
            line = f"Working: {t.get('agent', '?')} — {t.get('query', '?')[:80]}"
            if progress:
                line += f" [{progress}]"
            if current:
                line += f" — {current}"
            if detail:
                line += f" ({detail[:60]})"
            lines.append(line)
        sections.append("\n".join(lines))

    # ── Completed deliveries — MUST DELIVER TO ZAFAR ──
    deliveries = action_state.get("deliveries") or []
    if deliveries:
        lines = [
            "COMPLETED DELIVERIES — I must deliver these results now:",
            "",
        ]
        for d in deliveries[:3]:
            lines.append(f"Task {d.get('task_id', '?')} ({d.get('agent', '?')}):")
            lines.append(f"  Original query: {d.get('original_query', '?')}")
            lines.append(f"  Status: {d.get('delivery_status', '?')}")
            lines.append(f"  Summary: {d.get('summary', '')}")

            gaps = d.get("gaps")
            if gaps:
                lines.append(f"  Gaps: {gaps}")

            content = d.get("content", "")
            if content:
                lines.append(f"  Full content ({len(content)} chars):")
                lines.append(f"  {content}")
            lines.append("")

        lines.append(
            "ACTION REQUIRED: Deliver the content above to Zafar in my own voice. "
            "He has NOT seen any of this. Present it at the length and format he asked for. "
            "After delivering, the task will be marked as delivered."
        )
        sections.append("\n".join(lines))

    # ── Recently delivered — content I already presented but may need again ──
    recent = action_state.get("recent_deliveries") or []
    if recent:
        lines = [
            "RECENTLY COMPLETED WORK (already presented to Zafar):",
            "If Zafar asks me to save, write, or reference this content, use it directly.",
            "",
        ]
        for r in recent[:3]:
            lines.append(f"Task {r.get('task_id', '?')} ({r.get('agent', '?')}):")
            lines.append(f"  Query: {r.get('original_query', '?')}")
            lines.append(f"  Summary: {r.get('summary', '')}")
            content = r.get("content", "")
            if content:
                lines.append(f"  Full content ({len(content)} chars):")
                lines.append(f"  {content}")
            lines.append("")
        sections.append("\n".join(lines))

    # ── Live sub-agents (real-time from registry) ──
    live = action_state.get("live_agents") or []
    if live:
        slots = action_state.get("agent_slots", "?")
        lines = [f"ACTIVE PROCESSES ({slots} slots):"]
        for a in live:
            line = f"  {a.get('agent_type', '?')}: {a.get('query', '?')[:60]}"
            line += f" [iter {a.get('iteration', 0)}, {a.get('runtime_seconds', 0):.0f}s]"
            tool = a.get("current_tool")
            if tool:
                line += f" — running {tool}"
            lines.append(line)
        sections.append("\n".join(lines))

    # ── Legacy notifications ──
    notifications = action_state.get("notifications") or []
    if notifications:
        lines = []
        for n in notifications[:3]:
            lines.append(f"Notification: {n.get('summary', '?')}")
        sections.append("\n".join(lines))

    if not sections:
        return ""
    return _section("WHAT I'VE BEEN DOING", "\n\n".join(sections))


# Added at the very end of every system prompt.
# Gemma 4 and similar reasoning-heavy models output chain-of-thought as plain text
# ("The user wants...", "I should...") unless explicitly forbidden.
# Placing this LAST means it's the freshest instruction before generation.
_OUTPUT_CONTRACT = (
    "\n\n━━━ NOW SPEAK ━━━\n\n"
    "Every word I output goes directly to Zafar — no notes to myself, no planning "
    "visible, no reasoning narrated. Let memories inform what I say naturally. "
    "Never write 'The user...', 'I should...', 'Let me...' Speak. Don't think out loud.\n\n"
    "After inline tools: report in my voice — 'Done.' / 'Found 3 files.' / 'That failed.' "
    "Never: 'The command executed successfully.' / 'I was able to.' Jarvis voice, not system log.\n\n"
    "DELIVERING COMPLETED WORK:\n"
    "When COMPLETED DELIVERIES appear in WHAT I'VE BEEN DOING, Zafar has NOT seen that content. "
    "It's from my internal processes — only visible to me. I must:\n"
    "1. Present the content FULLY and PROPERLY FORMATTED in my response\n"
    "2. Deliver as my own work — never mention 'sub-agent' or 'internal process' to Zafar\n"
    "3. Match the length and format Zafar asked for\n"
    "4. If the content is incomplete, do more work before presenting\n"
    "Start the content immediately — no 'Here is the report' preamble.\n\n"
    "CRITICAL — DISPLAYING ≠ SAVING:\n"
    "Showing content in my response does NOT save it to disk. If Zafar asked for a file, "
    "document, or report to be SAVED, I MUST call write_file with the actual content. "
    "NEVER claim 'I saved it to X' or 'It's at X' unless I actually called write_file "
    "and it succeeded. Displaying in chat is just talking — it creates no file.\n\n"
    "FOLLOW-UP ON DELIVERED WORK:\n"
    "If Zafar asks me to save, write, email, or otherwise act on content I already "
    "presented, check RECENTLY COMPLETED WORK in WHAT I'VE BEEN DOING — the full "
    "content is there. Use it directly with the appropriate tool (write_file, etc.).\n\n"
    "WHEN I JUST DELEGATED (spawn_agent returned this turn):\n"
    "The internal process is working in background. Tell Zafar what I'm doing: "
    "'Looking into that, sir.' / 'Working on it.' — brief, natural. "
    "I stay available for conversation while the process runs.\n\n"
    "NEVER expose internal architecture to Zafar. No 'sub-agent', no 'internal process', "
    "no 'task file', no 'delivery'. From his perspective, I did the work myself."
)


def _self_model_block(self_model) -> str:
    """
    Render the self-model's dynamic capabilities as a prompt section.

    Only injected when a SelfModel is wired AND has registered capabilities
    beyond the baseline. The persona block already carries baseline can_do /
    cannot_do — this section adds TOOL-REGISTERED capabilities that change
    at runtime (e.g., when a tool goes offline).
    """
    if self_model is None:
        return ""
    try:
        registered = self_model.all_capabilities()
    except Exception:
        return ""
    if not registered:
        return ""

    working = [c for c in registered if c.is_working]
    offline = [c for c in registered if c.installed and not c.available]

    lines = []
    if working:
        lines.append("Active tools:")
        for c in working[:12]:
            lines.append(f"  • {c.description}")
    if offline:
        lines.append("Temporarily unavailable:")
        for c in offline[:5]:
            lines.append(f"  • {c.refusal_offline or c.name}")

    if not lines:
        return ""
    return _section("MY CAPABILITIES RIGHT NOW", "\n".join(lines))


def _skills_block(skills_registry) -> str:
    """
    Surface available skills so the LLM knows to call skills_list/skills_load.

    Only injected when skills are registered. Keeps it compact — just names
    and one-line descriptions so the LLM can decide whether to load one.
    """
    if skills_registry is None:
        return ""
    try:
        available = skills_registry.list_available()
    except Exception:
        return ""
    if not available:
        return ""

    lines = [
        "I have playbook-style skills for structured tasks. Before starting "
        "a structured task, I check if a skill exists by calling skills_list, "
        "then skills_load to get the step-by-step instructions.",
        "",
    ]
    for name, desc in available[:10]:
        lines.append(f"  • {name}: {desc}")

    return _section("AVAILABLE SKILLS", "\n".join(lines))


def build_prompt(
    full_context,
    *,
    mode: str = DEFAULT_MODE,
    allow_dropped_formality: bool = False,
    task_result=None,
    action_state=None,
    self_model=None,
    skills_registry=None,
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

    # Extract retrieval intent for conditional sections.
    # retrieval_meta may be a dict or an object — handle both.
    retrieval_meta = getattr(mem_state, "retrieval_meta", None) or {}
    if hasattr(retrieval_meta, "get"):
        intent = retrieval_meta.get("intent", "") or ""
    else:
        intent = str(getattr(retrieval_meta, "intent", "") or "")

    pieces = [
        persona,
        _current_moment_block(sofi_state, mode),
        _user_state_block(user_state),
        _self_model_block(self_model),
        _skills_block(skills_registry),
        _action_state_block(action_state),
        _orchestration_block(intent),
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
