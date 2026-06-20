"""
BRAIN/state/user_state.py — User State Inferencer

Reads existing signals from WorkingContext + the raw message and produces a
rule-based snapshot of Zafar's current state. Zero LLM calls, ~1ms per turn.

The output gets written into WorkingContext.user via context_manager.update_user_state
so the prompt builder surfaces it under WHAT'S TRUE FOR ZAFAR.

The output also feeds the mode controller as one of several signals.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


# ============================================================================
# OUTPUT TYPES
# ============================================================================

# Possible emotional state labels — small fixed vocabulary
EMOTIONAL_STATES = (
    "neutral", "stressed", "sad", "frustrated", "overwhelmed",
    "excited", "content", "focused", "tired",
)

NEEDS = ("emotional_support", "practical", "informational", "creative", "casual")

ENGAGEMENT_LEVELS = ("disengaged", "normal", "highly_engaged")


@dataclass
class UserStateUpdate:
    """Result of one inference pass — directly applied to WorkingContext.user."""
    current_emotional_state: str = "neutral"
    emotional_intensity: float = 0.0
    current_need: str = "casual"
    engagement_level: str = "normal"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "current_emotional_state": self.current_emotional_state,
            "emotional_intensity": self.emotional_intensity,
            "current_need": self.current_need,
            "engagement_level": self.engagement_level,
        }


# ============================================================================
# LEXICAL PATTERNS (pre-compiled at import time)
# ============================================================================

_EMOTION_PATTERNS = {
    "stressed":     re.compile(r"\b(stressed?|stress|pressure|anxious|anxiety|worried|nervous)\b", re.I),
    "sad":          re.compile(r"\b(sad|down|depressed|low|miserable|grief|grieving|crying|cried)\b", re.I),
    "frustrated":   re.compile(r"\b(frustrat\w+|annoy\w+|angry|pissed|irritated|fed up|sick of)\b", re.I),
    "overwhelmed":  re.compile(r"\b(overwhelm\w+|drowning|too much|can'?t handle|burnt out|burning out|breaking)\b", re.I),
    "excited":      re.compile(r"\b(excited|stoked|hyped|thrilled|amazing|incredible|love it)\b", re.I),
    "content":      re.compile(r"\b(good|fine|alright|okay|happy|content|chill)\b", re.I),
    "tired":        re.compile(r"\b(tired|exhausted|wiped|drained|sleepy|no energy|done with)\b", re.I),
}

# Words that strongly signal informational/learning need
_INFO_PATTERN = re.compile(
    r"\b(what|why|how|when|where|which|explain|describe|tell me about|"
    r"define|meaning of|difference between|compared to)\b",
    re.I,
)

# Words that strongly signal practical / task-oriented need
_PRACTICAL_PATTERN = re.compile(
    r"\b(fix|debug|implement|write|build|make|create|run|deploy|"
    r"refactor|change|update|add|remove|test|check|review)\b",
    re.I,
)

# Words that signal creative work
_CREATIVE_PATTERN = re.compile(
    r"\b(brainstorm|design|imagine|draft|sketch|come up with|"
    r"poem|story|essay|name for|idea for|creative)\b",
    re.I,
)

# Casual / chit-chat markers
_CASUAL_PATTERN = re.compile(
    r"^(hi|hey|hello|yo|sup|hii|hii+|how'?s it|what'?s up|just wanted to|just saying)\b",
    re.I,
)

# Strong intensity multipliers — punctuation density, capitalization, hedging
_INTENSITY_AMP_PUNCT = re.compile(r"[!?]{2,}|\.\.\.")
_INTENSITY_AMP_CAPS = re.compile(r"\b[A-Z]{3,}\b")  # SHOUTING

# Engagement signals
_HIGH_ENGAGEMENT_KEYWORDS = re.compile(
    r"\b(lets|let'?s|let me|can we|i want to|i'?m gonna|i need to|next|so what about)\b",
    re.I,
)
_LOW_ENGAGEMENT_KEYWORDS = re.compile(
    r"^(ok|okay|sure|right|fine|whatever|i guess|alright)\.?\s*$",
    re.I,
)


# ============================================================================
# INFERENCER
# ============================================================================

class UserStateInferencer:
    """Rule-based inferencer. Stateless; safe to call from any thread."""

    def infer(
        self,
        ctx: Any,
        message: str,
        prev_state: Optional[UserStateUpdate] = None,
    ) -> UserStateUpdate:
        """
        Args:
            ctx:        WorkingContext snapshot from MemoryManager.get_full_context().
            message:    Raw user message text.
            prev_state: Previous turn's inferred state, for smoothing.

        Returns:
            UserStateUpdate ready to be written to WorkingContext.user.
        """
        msg = (message or "").strip()
        msg_l = msg.lower()
        n_words = len(msg.split())

        intent = self._safe_intent(ctx)
        memory_emotional_baseline = self._safe_emotional_baseline(ctx)

        # 1. EMOTIONAL STATE — keyword detection over message + memory tone
        emotion, intensity = self._detect_emotion(
            msg, msg_l, intent, memory_emotional_baseline,
        )

        # 2. NEED — derived from intent + emotion + message phrasing
        need = self._infer_need(msg, msg_l, intent, emotion, intensity)

        # 3. ENGAGEMENT — from message length, tempo, lexical markers
        engagement = self._engagement(msg, msg_l, n_words)

        # 4. Smoothing / decay — emotional moments don't disappear in one turn.
        # Intensity decays 0.7× per turn when the prev turn was at least 0.25.
        # The LABEL is more conservative — it only carries forward when the
        # prev turn was meaningfully emotional (>= 0.40), so we don't keep
        # showing "content" or "stressed" on plainly-neutral follow-ups.
        if prev_state and prev_state.emotional_intensity >= 0.25:
            floor = prev_state.emotional_intensity * 0.7
            intensity = max(intensity, floor)
            # Only carry the LABEL when the prev turn was clearly emotional —
            # avoids "content" sticking across multiple neutral follow-ups.
            if (
                emotion == "neutral"
                and prev_state.current_emotional_state != "neutral"
                and prev_state.emotional_intensity >= 0.40
            ):
                emotion = prev_state.current_emotional_state

        return UserStateUpdate(
            current_emotional_state=emotion,
            emotional_intensity=round(intensity, 2),
            current_need=need,
            engagement_level=engagement,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_intent(ctx) -> str:
        try:
            meta = ctx.memory.retrieval_meta or {}
            return (meta.get("intent") or "ambient")
        except Exception:
            return "ambient"

    @staticmethod
    def _safe_emotional_baseline(ctx) -> Dict[str, Any]:
        try:
            return ctx.memory.emotional_baseline or {}
        except Exception:
            return {}

    def _detect_emotion(
        self,
        msg: str,
        msg_l: str,
        intent: str,
        memory_baseline: Dict[str, Any],
    ):
        """
        Returns (emotion_label, intensity in [0, 1]).

        Multi-signal additive scoring. Memory baseline + intent + punctuation
        are PRIMARY signals (language-agnostic). English keyword patterns are
        a TERTIARY hint — useful when they match, silent when they don't.
        Designed not to over-rely on a hardcoded vocabulary that can never
        cover every way a person can express emotion.
        """
        intensity = 0.0

        # PRIMARY 1: Memory-side baseline — the router already weighed
        # whether what's surfaced is emotionally heavy. Use that directly.
        avg_tone = memory_baseline.get("avg_emotional_tone")
        if isinstance(avg_tone, (int, float)):
            intensity += min(abs(avg_tone) * 0.4, 0.40)

        # PRIMARY 2: Intent — if the memory router classified this turn as
        # emotional, that's a strong signal regardless of keywords.
        if intent == "emotional":
            intensity += 0.40

        # PRIMARY 3: Punctuation density and ALL-CAPS — language-agnostic.
        # These get a real floor instead of needing a keyword to amplify.
        if _INTENSITY_AMP_PUNCT.search(msg):
            intensity += 0.25
        if _INTENSITY_AMP_CAPS.search(msg):
            intensity += 0.20

        # TERTIARY: English keyword hint — small boost when patterns match.
        # The patterns can never cover everything (Hinglish, new slang,
        # idioms we haven't listed), so they shouldn't be the foundation.
        keyword_hits = [
            label for label, pat in _EMOTION_PATTERNS.items()
            if pat.search(msg_l)
        ]
        if keyword_hits:
            intensity += 0.20
            if len(keyword_hits) >= 2:
                intensity += 0.10

        intensity = min(intensity, 1.0)

        # ── Emotion LABEL — derived from intensity + memory baseline + intent ──
        # Use English keyword hits when they give a specific label, otherwise
        # pick from a small set based on the primary signals.
        if keyword_hits:
            priority = ["overwhelmed", "stressed", "frustrated", "sad",
                        "tired", "excited", "content"]
            emotion = next((p for p in priority if p in keyword_hits),
                           keyword_hits[0])
        elif isinstance(avg_tone, (int, float)) and avg_tone < -0.4:
            # Memory tone is heavily negative — surface that
            emotion = "sad"
        elif intent == "emotional" and intensity >= 0.4:
            # Router said emotional, no specific label — generic
            emotion = "stressed"
        elif isinstance(avg_tone, (int, float)) and avg_tone > 0.4 and intensity < 0.3:
            emotion = "content"
        else:
            emotion = "neutral"

        return emotion, intensity

    def _infer_need(
        self,
        msg: str,
        msg_l: str,
        intent: str,
        emotion: str,
        intensity: float,
    ) -> str:
        """High-confidence: emotional intensity ≥ 0.5 → emotional_support."""
        if intensity >= 0.5 or emotion in ("sad", "overwhelmed", "frustrated"):
            return "emotional_support"

        if _CREATIVE_PATTERN.search(msg_l):
            return "creative"
        if _PRACTICAL_PATTERN.search(msg_l):
            return "practical"
        if _INFO_PATTERN.search(msg_l) or intent == "factual":
            return "informational"
        if _CASUAL_PATTERN.search(msg_l) or intent == "ambient":
            return "casual"

        # Soft fallback by intent
        if intent == "emotional":
            return "emotional_support"
        if intent in ("entity", "temporal"):
            return "informational"
        return "casual"

    def _engagement(self, msg: str, msg_l: str, n_words: int) -> str:
        if n_words <= 3 and _LOW_ENGAGEMENT_KEYWORDS.match(msg_l):
            return "disengaged"
        if _HIGH_ENGAGEMENT_KEYWORDS.search(msg_l):
            return "highly_engaged"
        if n_words > 25:
            return "highly_engaged"
        return "normal"
