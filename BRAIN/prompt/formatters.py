"""
BRAIN/prompt/formatters.py — Memory dict → readable line

Turns raw memory dicts from MemoryRetrievalEngine into compact, natural
single-line sentences for the prompt. Strips internal fields, adds light
temporal and emotional qualifiers only when meaningful.

Used by builder.py.
"""

from datetime import datetime
from typing import Dict, List, Optional

# Fields we never surface — internal scores, IDs, embeddings, etc.
_INTERNAL_KEYS = frozenset({
    "id", "root_id", "_trigger_entity", "_tier_hint", "_coverage_source",
    "_recency_boosted", "bm25_score", "score", "activation_score",
    "edge_strength", "path_count", "distance", "access_count",
    "last_accessed", "content_vector", "rel_type",
})

# Hard cap on per-memory content length (chars). Prevents a single verbose
# memory from blowing the token budget.
_MAX_CONTENT_CHARS = 200


def _parse_ts(raw) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", ""))
    except Exception:
        return None


def _age_qualifier(ts_raw, now: Optional[datetime] = None) -> str:
    """Return a short relative-time tag like '(3 days ago)' or '' if recent."""
    dt = _parse_ts(ts_raw)
    if dt is None:
        return ""
    now = now or datetime.now()
    days = (now - dt.replace(tzinfo=None)).days
    if days < 2:
        return ""
    if days < 7:
        return f"({days} days ago)"
    if days < 30:
        weeks = days // 7
        return f"({weeks} week{'s' if weeks > 1 else ''} ago)"
    months = days // 30
    return f"({months} month{'s' if months > 1 else ''} ago)"


def _emotion_qualifier(tone) -> str:
    """Return a brief affect note if the memory carries strong emotion."""
    try:
        t = float(tone or 0.0)
    except (TypeError, ValueError):
        return ""
    if abs(t) < 0.4:
        return ""
    if t > 0:
        return "[felt positively]"
    return "[felt heavily]"


def format_memory(m: Dict) -> str:
    """
    Format one memory dict into a readable single-line bullet.

    Examples:
      - Sarah is a colleague who tends to ramble in conversations.
      - User scheduled a coffee with Sarah for Thursday 8am at Roasters. (3 days ago)
      - Project deadline was pushed to next Friday. (1 week ago) [felt heavily]
    """
    # Choose the most identifying content field available.
    content = (
        m.get("content")
        or m.get("root_content")
        or m.get("description")
        or ""
    ).strip()

    # If a named entity is present, lead with it (more useful for the model).
    name = m.get("person_name") or m.get("concept")
    if name and name.lower() not in content.lower():
        content = f"{name}: {content}" if content else name

    # Hard cap — prevents verbose memories from blowing the token budget.
    if len(content) > _MAX_CONTENT_CHARS:
        content = content[:_MAX_CONTENT_CHARS - 3].rsplit(" ", 1)[0] + "..."

    # Clean trailing punctuation duplication
    content = content.rstrip(".")

    age = _age_qualifier(m.get("timestamp") or m.get("root_timestamp"))
    emo = _emotion_qualifier(m.get("emotional_tone"))

    tail_parts = [p for p in (age, emo) if p]
    tail = " " + " ".join(tail_parts) if tail_parts else ""

    return f"- {content}.{tail}"


def format_memories(memories: List[Dict], max_items: int = 10) -> str:
    """Format a list of memories into a newline-joined block. Caps at max_items."""
    if not memories:
        return ""
    return "\n".join(format_memory(m) for m in memories[:max_items])
