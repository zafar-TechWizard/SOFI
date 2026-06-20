"""
BRAIN/mode/signals.py — Lexical signal extraction for the mode controller

Pre-compiled regex patterns that detect category-of-discourse signals from
the raw message. Pure function calls — no state, ~0.1ms.

Used by controller.py as one input to mode scoring.
"""

import re


# ============================================================================
# Lexical patterns
# ============================================================================

# TECHNICAL — code, debugging, infrastructure, system internals
_TECHNICAL_RE = re.compile(
    r"\b(code|debug|debugging|error|stack ?trace|exception|function|class|"
    r"method|api|endpoint|sql|query|schema|database|index|cache|deploy|"
    r"deployment|build|compile|merge|rebase|commit|push|pull request|pr|"
    r"refactor|test|unittest|pytest|module|package|import|env|config|"
    r"docker|kubernetes|k8s|terraform|ci|cd|pipeline|server|backend|"
    r"frontend|react|python|javascript|typescript|rust|go(?:lang)?|cpp|"
    r"sdk|library|framework|model|inference|gradient|tensor|neural|"
    r"latency|throughput|memory leak|race condition|deadlock|mutex)\b",
    re.I,
)

# CREATIVE — brainstorming, design, writing, imagination
_CREATIVE_RE = re.compile(
    r"\b(brainstorm|brainstorming|design|sketch|imagine|come up with|"
    r"draft|outline|name for|idea for|let'?s create|let'?s build|"
    r"poem|story|essay|article|blog|caption|tagline|slogan|metaphor|"
    r"creative|invent|dream up|reimagine|fictional|concept for)\b",
    re.I,
)

# EMPATHY — emotional or supportive context
_EMPATHY_RE = re.compile(
    r"\b(feel|feeling|felt|stressed?|overwhelmed?|burnt out|exhausted|"
    r"sad|down|depressed|low|tired of|sick of|frustrated|angry|hurt|"
    r"can'?t handle|can'?t do this|breaking|cried|crying|grief|grieving|"
    r"lonely|alone|miss|losing|lost|broken|empty|drained|done with)\b",
    re.I,
)

# PLAYFUL — casual chatter, jokes, banter
_PLAYFUL_RE = re.compile(
    r"\b(lol|haha+|lmao|rofl|hehe|jk|btw|tbh|imo|imho|ngl|fr|"
    r"hii+|hey there|sup|yo)\b",
    re.I,
)

# CODE BLOCK — triple backticks (strong focused signal)
_CODE_BLOCK_RE = re.compile(r"```")

# IMPERATIVE — "write me X", "fix this", direct command
_IMPERATIVE_RE = re.compile(
    r"^(write|fix|make|build|implement|generate|create|refactor|"
    r"explain|summarize|summarise|translate|convert|optimize|optimise|"
    r"check|review|find|search|list|count|compare|debug|test|run)\b",
    re.I,
)

# QUESTION — informational-style question words. Boosts focused.
# Note: deliberately narrow (anchored to "what/how/why/when/where + verb")
# to avoid matching every casual "what's up".
_QUESTION_RE = re.compile(
    r"\b(what (is|are|was|were|does|do|did|happens|happened|caused|"
    r"makes|means|kind of)|how (do|does|did|can|to|much|many|long|"
    r"often)|why (is|are|does|do|did|am|would|should)|"
    r"when (is|was|did|does|do)|where (is|was|did|does|do))\b",
    re.I,
)

# EXPLICIT CREATIVE TRIGGERS — strong enough to be a hard override
_EXPLICIT_CREATIVE_RE = re.compile(
    r"\b(brainstorm with me|help me design|help me imagine|"
    r"come up with names|let'?s design|let'?s imagine|"
    r"give me ideas for|creative angle|creative spin)\b",
    re.I,
)


# ============================================================================
# Public API
# ============================================================================

class SignalProfile:
    """Lightweight container for lexical signal hits on a message."""

    __slots__ = (
        "technical", "creative", "empathy", "playful",
        "has_code_block", "is_imperative", "explicit_creative",
        "is_question", "word_count",
    )

    def __init__(
        self,
        technical: bool = False,
        creative: bool = False,
        empathy: bool = False,
        playful: bool = False,
        has_code_block: bool = False,
        is_imperative: bool = False,
        explicit_creative: bool = False,
        is_question: bool = False,
        word_count: int = 0,
    ):
        self.technical = technical
        self.creative = creative
        self.empathy = empathy
        self.playful = playful
        self.has_code_block = has_code_block
        self.is_imperative = is_imperative
        self.explicit_creative = explicit_creative
        self.is_question = is_question
        self.word_count = word_count

    def __repr__(self) -> str:
        flags = [
            f for f in (
                "technical", "creative", "empathy", "playful",
                "has_code_block", "is_imperative", "explicit_creative",
                "is_question",
            ) if getattr(self, f)
        ]
        return f"SignalProfile({', '.join(flags) or 'none'}, n={self.word_count})"


def extract_signals(message: str) -> SignalProfile:
    """
    Run every pre-compiled regex over the message exactly once.
    Returns a SignalProfile with booleans for each category + word count.
    """
    msg = (message or "")
    msg_lower = msg.lower()

    return SignalProfile(
        technical=bool(_TECHNICAL_RE.search(msg_lower)),
        creative=bool(_CREATIVE_RE.search(msg_lower)),
        empathy=bool(_EMPATHY_RE.search(msg_lower)),
        playful=bool(_PLAYFUL_RE.search(msg_lower)),
        has_code_block=bool(_CODE_BLOCK_RE.search(msg)),
        is_imperative=bool(_IMPERATIVE_RE.match(msg.strip())),
        explicit_creative=bool(_EXPLICIT_CREATIVE_RE.search(msg_lower)),
        is_question=bool(_QUESTION_RE.search(msg_lower)),
        word_count=len(msg.split()),
    )
