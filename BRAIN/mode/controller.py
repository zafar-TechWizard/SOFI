"""
BRAIN/mode/controller.py — Mode Controller

Non-LLM, signal-scored, hysteretic decision over 4 modes:
  conversational | empathetic | focused | creative

Design principles:
  1. Multi-signal additive scoring — no single source decides except hard overrides
  2. Calibrated weights — encoded as named constants, easy to tune
  3. Hysteresis — prev_mode gets a small "stay" bias (+0.20)
  4. Margin gate — winner must beat second by ≥ 0.15 to switch; otherwise hold
  5. Hard overrides — high emotional intensity / explicit creative trigger /
                       code block in message all bypass scoring
  6. Default to conversational — if every score is near zero, fall back to the
                                  warm baseline rather than holding stale mode

Total cost: ~0.1ms per turn. Deterministic. Fully debuggable.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from BRAIN.mode.signals import SignalProfile, extract_signals


# ============================================================================
# MODE ENUM
# ============================================================================

class Mode(str, Enum):
    CONVERSATIONAL = "conversational"
    EMPATHETIC = "empathetic"
    FOCUSED = "focused"
    CREATIVE = "creative"


DEFAULT_MODE = Mode.CONVERSATIONAL
ALL_MODES = tuple(Mode)


# ============================================================================
# CALIBRATED WEIGHTS  (named so they're easy to tune)
# ============================================================================

# A. Intent bias — strong signal from memory router
W_INTENT_EMOTIONAL_TO_EMPATHETIC   = 0.60
W_INTENT_FACTUAL_TO_FOCUSED        = 0.40
W_INTENT_ENTITY_TO_CONVERSATIONAL  = 0.30
W_INTENT_TEMPORAL_TO_CONVERSATIONAL = 0.25
W_INTENT_AMBIENT_TO_CONVERSATIONAL = 0.40

# B. User need bias — strong signal from state inferencer
W_NEED_EMOTIONAL_SUPPORT = 0.50
W_NEED_INFORMATIONAL     = 0.45
W_NEED_CASUAL            = 0.50
W_NEED_CREATIVE          = 0.55
W_NEED_PRACTICAL         = 0.35

# C. Emotional intensity bands
W_INTENSITY_MID  = 0.20   # 0.30 <= intensity < 0.50
W_INTENSITY_HIGH = 0.40   # 0.50 <= intensity < 0.70
THRESH_INTENSITY_OVERRIDE = 0.70  # >= triggers hard override → empathetic

# D. Lexical signals (regex hits from BRAIN/mode/signals.py)
W_LEX_TECHNICAL  = 0.45
W_LEX_CREATIVE   = 0.50
W_LEX_EMPATHY    = 0.35
W_LEX_PLAYFUL    = 0.30
W_LEX_CODE_BLOCK = 0.40   # also a hard override
W_LEX_IMPERATIVE = 0.30
W_LEX_QUESTION   = 0.30   # "what is X" / "how does Y" → focused

# E. Message shape (light signals)
W_SHAPE_SHORT_CASUAL = 0.20  # ≤ 3 words, no question
W_SHAPE_LONG_TECH    = 0.20  # > 50 words AND technical hit
W_SHAPE_MULTI_EMO    = 0.25  # multiple emotional indicators

# F. Hysteresis — small "stay" bias toward previous mode
W_HYSTERESIS = 0.20
# When the previous mode was set by a HARD OVERRIDE (intensity override,
# explicit creative phrase, code block), the user has signaled strong
# commitment to that mode. Give the next turn more room to keep the mode
# rather than snapping back to conversational on a vague follow-up.
W_HYSTERESIS_AFTER_OVERRIDE = 1.00

# Confidence gate — winner must beat 2nd-place by this much, else hold prev
MIN_MARGIN_FOR_SWITCH = 0.15


# ============================================================================
# OUTPUT TYPE
# ============================================================================

@dataclass
class ModeDecision:
    """Output of one controller pass."""
    mode: Mode
    allow_dropped_formality: bool
    scores: Dict[Mode, float] = field(default_factory=dict)
    triggered_overrides: list = field(default_factory=list)
    held_prev: bool = False  # True if margin gate prevented a switch

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "allow_dropped_formality": self.allow_dropped_formality,
            "scores": {m.value: round(v, 3) for m, v in self.scores.items()},
            "triggered_overrides": list(self.triggered_overrides),
            "held_prev": self.held_prev,
        }


# ============================================================================
# CONTROLLER
# ============================================================================

class ModeController:
    """Stateless. Pass prev_mode each call for hysteresis."""

    def decide(
        self,
        ctx: Any,
        message: str,
        prev_mode: Mode = DEFAULT_MODE,
        prev_was_override: bool = False,
    ) -> ModeDecision:
        """
        Decide which mode SOFi should be in for this turn.

        Args:
            ctx:      WorkingContext snapshot from MemoryManager.get_full_context().
                      Reads from ctx.user (must already be filled by UserStateInferencer)
                      and ctx.memory.retrieval_meta.
            message:  Raw user message text.
            prev_mode: The mode used on the previous turn (for hysteresis).

        Returns:
            ModeDecision — selected mode + the allow_dropped_formality flag.
        """
        sig = extract_signals(message)
        intent = self._safe_intent(ctx)
        need = self._safe_need(ctx)
        intensity = self._safe_intensity(ctx)

        # ── Phase 1: HARD OVERRIDES (skip scoring) ───────────────────────────
        triggered = []

        if intensity >= THRESH_INTENSITY_OVERRIDE:
            triggered.append("intensity_override")
            return self._finalize(
                Mode.EMPATHETIC, intensity, prev_mode,
                scores={}, triggered=triggered, held_prev=False,
            )

        if sig.explicit_creative:
            triggered.append("explicit_creative_phrase")
            return self._finalize(
                Mode.CREATIVE, intensity, prev_mode,
                scores={}, triggered=triggered, held_prev=False,
            )

        if sig.has_code_block:
            triggered.append("code_block_present")
            return self._finalize(
                Mode.FOCUSED, intensity, prev_mode,
                scores={}, triggered=triggered, held_prev=False,
            )

        # ── Phase 2: Score every mode ────────────────────────────────────────
        scores: Dict[Mode, float] = {m: 0.0 for m in ALL_MODES}

        # A — Intent bias
        if intent == "emotional":
            scores[Mode.EMPATHETIC] += W_INTENT_EMOTIONAL_TO_EMPATHETIC
        elif intent == "factual":
            scores[Mode.FOCUSED] += W_INTENT_FACTUAL_TO_FOCUSED
        elif intent == "entity":
            scores[Mode.CONVERSATIONAL] += W_INTENT_ENTITY_TO_CONVERSATIONAL
        elif intent == "temporal":
            scores[Mode.CONVERSATIONAL] += W_INTENT_TEMPORAL_TO_CONVERSATIONAL
        elif intent == "ambient":
            scores[Mode.CONVERSATIONAL] += W_INTENT_AMBIENT_TO_CONVERSATIONAL

        # B — Need bias
        if need == "emotional_support":
            scores[Mode.EMPATHETIC] += W_NEED_EMOTIONAL_SUPPORT
        elif need == "informational":
            scores[Mode.FOCUSED] += W_NEED_INFORMATIONAL
        elif need == "casual":
            scores[Mode.CONVERSATIONAL] += W_NEED_CASUAL
        elif need == "creative":
            scores[Mode.CREATIVE] += W_NEED_CREATIVE
        elif need == "practical":
            scores[Mode.FOCUSED] += W_NEED_PRACTICAL

        # C — Emotional intensity bands
        if intensity >= 0.50:
            scores[Mode.EMPATHETIC] += W_INTENSITY_HIGH
        elif intensity >= 0.30:
            scores[Mode.EMPATHETIC] += W_INTENSITY_MID

        # D — Lexical signals
        if sig.technical:
            scores[Mode.FOCUSED] += W_LEX_TECHNICAL
        if sig.creative:
            scores[Mode.CREATIVE] += W_LEX_CREATIVE
        if sig.empathy:
            scores[Mode.EMPATHETIC] += W_LEX_EMPATHY
        if sig.playful:
            scores[Mode.CONVERSATIONAL] += W_LEX_PLAYFUL
        if sig.is_imperative:
            scores[Mode.FOCUSED] += W_LEX_IMPERATIVE
        if sig.is_question:
            scores[Mode.FOCUSED] += W_LEX_QUESTION

        # E — Shape signals
        if sig.word_count <= 3 and not sig.empathy and not sig.technical:
            scores[Mode.CONVERSATIONAL] += W_SHAPE_SHORT_CASUAL
        if sig.word_count > 50 and sig.technical:
            scores[Mode.FOCUSED] += W_SHAPE_LONG_TECH
        if sig.empathy and intensity >= 0.30:
            # Empathy keyword + measurable intensity = stronger empathetic pull
            scores[Mode.EMPATHETIC] += W_SHAPE_MULTI_EMO

        # F — Hysteresis: previous mode gets a "stay" bonus.
        # If the previous mode was set by a HARD override, the user has
        # committed strongly to that mode — give it more staying power on
        # a vague follow-up so we don't snap back to conversational.
        scores[prev_mode] += (
            W_HYSTERESIS_AFTER_OVERRIDE if prev_was_override else W_HYSTERESIS
        )

        # ── Phase 3: Pick winner with margin gate ────────────────────────────
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        winner, winner_score = ranked[0]
        runner, runner_score = ranked[1] if len(ranked) > 1 else (winner, 0.0)

        # All zero (no signal fired anywhere) → conversational, NOT prev_mode.
        # Prev_mode might be stale empathetic; warm baseline is safer.
        if winner_score <= 0.0:
            triggered.append("no_signal_default_conversational")
            return self._finalize(
                Mode.CONVERSATIONAL, intensity, prev_mode,
                scores=scores, triggered=triggered, held_prev=False,
            )

        # Margin gate — winner must clearly beat second place.
        # Use a small epsilon to avoid float-precision bites: 0.75 - 0.60
        # evaluates to 0.14999999... in IEEE 754, which is technically less
        # than 0.15 but represents a real 0.15 gap.
        held_prev = False
        gap = round(winner_score - runner_score, 6)
        if winner != prev_mode and gap < MIN_MARGIN_FOR_SWITCH:
            triggered.append(f"margin_too_thin_held_{prev_mode.value}")
            winner = prev_mode
            held_prev = True

        return self._finalize(
            winner, intensity, prev_mode,
            scores=scores, triggered=triggered, held_prev=held_prev,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize(
        mode: Mode,
        intensity: float,
        prev_mode: Mode,
        scores: Dict[Mode, float],
        triggered: list,
        held_prev: bool,
    ) -> ModeDecision:
        # Earned exception flag — empathetic + measurable intensity unlocks
        # the dropped-formality permission in the persona prompt.
        allow_drop = mode == Mode.EMPATHETIC and intensity >= 0.60
        return ModeDecision(
            mode=mode,
            allow_dropped_formality=allow_drop,
            scores=scores,
            triggered_overrides=triggered,
            held_prev=held_prev,
        )

    @staticmethod
    def _safe_intent(ctx) -> str:
        try:
            return (ctx.memory.retrieval_meta or {}).get("intent", "ambient")
        except Exception:
            return "ambient"

    @staticmethod
    def _safe_need(ctx) -> str:
        try:
            return getattr(ctx.user, "current_need", None) or "casual"
        except Exception:
            return "casual"

    @staticmethod
    def _safe_intensity(ctx) -> float:
        try:
            v = getattr(ctx.user, "emotional_intensity", 0.0)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0
