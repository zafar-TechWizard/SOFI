"""
BRAIN/state/response_analyzer.py — Analyze SOFi's own responses.

Extracts three lightweight signals from each assistant response:
  - topics discussed (what subjects came up)
  - commitments made ("I'll look into that", "let me check")
  - questions asked (SOFi-initiated questions back to Zafar)

These feed into SofiState.last_* fields so the next turn's prompt can
include a compact "WHAT I SAID LAST TURN" section — giving SOFi memory
of her own side of the dialogue without an LLM call.

All extraction is regex-based, < 1ms, no LLM. Runs as a post-response
fire-and-forget task so it never delays streaming.
"""

import re
from dataclasses import dataclass
from typing import List


# ── Topic extraction ─────────────────────────────────────────────────────────
# Catch: file paths, URLs, code identifiers, proper nouns, quoted terms.
# Deliberately broad — false positives are harmless here; the prompt builder
# caps output at 5 items anyway.

_FILE_PATH = re.compile(r'(?:^|[\s`\'"])(/?[\w.-]+/[\w./-]+|[\w.-]+\.(?:py|js|ts|json|yaml|yml|md|txt|sh|html|css|sql))', re.I)
_QUOTED = re.compile(r'[`"\']([^`"\']{3,60})[`"\'']')
_CODE_IDENT = re.compile(r'\b([A-Z][a-z]+[A-Z]\w*|[a-z]+_[a-z_]+)\b')  # CamelCase or snake_case
_PROPER_NOUN = re.compile(r'(?<!\.\s)\b([A-Z][a-z]{2,}\b(?:\s+[A-Z][a-z]{2,}\b)*)')

# Stop-list: words to never emit as topics
_STOPWORDS = frozenset({
    "Sir", "Zafar", "Yes", "No", "Let", "The", "That", "This", "Here",
    "Done", "Noted", "Okay", "Right", "Sure", "Well", "Also", "Just",
    "Good", "Great", "One", "Two", "But", "And", "For", "With", "From",
})


def extract_topics(text: str) -> List[str]:
    """
    Return up to 5 topic tokens from SOFi's response.

    Extraction priority: file paths > quoted terms > code identifiers > proper nouns.
    """
    seen: set = set()
    results: List[str] = []

    def _add(item: str) -> None:
        clean = item.strip().strip("`\"'")
        if clean and clean not in seen and clean not in _STOPWORDS and len(clean) >= 3:
            seen.add(clean)
            results.append(clean)

    for m in _FILE_PATH.finditer(text):
        _add(m.group(1))
        if len(results) >= 5:
            return results

    for m in _QUOTED.finditer(text):
        _add(m.group(1))
        if len(results) >= 5:
            return results

    for m in _CODE_IDENT.finditer(text):
        _add(m.group(1))
        if len(results) >= 5:
            return results

    for m in _PROPER_NOUN.finditer(text):
        _add(m.group(1))
        if len(results) >= 5:
            return results

    return results


# ── Commitment extraction ─────────────────────────────────────────────────────
# Patterns that signal SOFi has undertaken something.

_COMMITMENT_PATTERNS = [
    re.compile(r"I'?ll\s+(.{5,80}?)(?:\.|,|$)", re.I | re.M),
    re.compile(r"I will\s+(.{5,80}?)(?:\.|,|$)", re.I | re.M),
    re.compile(r"let me\s+(.{5,80}?)(?:\.|,|$)", re.I | re.M),
    re.compile(r"I can\s+(?:look into|check|find|help with)\s+(.{5,80}?)(?:\.|,|$)", re.I | re.M),
    re.compile(r"I'?ll\s+(?:look into|check|find|help with)\s+(.{5,80}?)(?:\.|,|$)", re.I | re.M),
    re.compile(r"I'?m\s+(?:going to|on it|working on)\s+(.{5,80}?)(?:\.|,|$)", re.I | re.M),
]


def extract_commitments(text: str) -> List[str]:
    """Return up to 3 commitment phrases from SOFi's response."""
    seen: set = set()
    results: List[str] = []

    for pattern in _COMMITMENT_PATTERNS:
        for m in pattern.finditer(text):
            phrase = m.group(0).strip().rstrip(".,")
            # Deduplicate by first 30 chars of the full match
            key = phrase[:30].lower()
            if key not in seen and len(phrase) >= 8:
                seen.add(key)
                results.append(phrase)
                if len(results) >= 3:
                    return results

    return results


# ── Question extraction ───────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def extract_questions(text: str) -> List[str]:
    """Return up to 2 questions SOFi asked in her response."""
    sentences = _SENTENCE_SPLIT.split(text)
    questions: List[str] = []
    for s in sentences:
        s = s.strip()
        if s.endswith("?") and len(s) >= 10:
            questions.append(s)
            if len(questions) >= 2:
                break
    return questions


# ── Convenience wrapper ───────────────────────────────────────────────────────

@dataclass
class ResponseAnalysis:
    topics: List[str]
    commitments: List[str]
    questions: List[str]


class ResponseAnalyzer:
    """
    Stateless analyzer for SOFi's own responses.

    Usage:
        analyzer = ResponseAnalyzer()
        analysis = analyzer.analyze(response_text)
        # analysis.topics, analysis.commitments, analysis.questions
    """

    def analyze(self, text: str) -> ResponseAnalysis:
        return ResponseAnalysis(
            topics=extract_topics(text),
            commitments=extract_commitments(text),
            questions=extract_questions(text),
        )
