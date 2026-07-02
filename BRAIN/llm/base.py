"""
BRAIN/llm/base.py — LLM Provider Protocol

Formalizes the contract that both GroqClient and GeminiClient already
satisfy. Using Protocol (structural subtyping) instead of ABC so
existing clients conform without inheritance changes.

Future providers (Ollama, Claude, OpenAI) implement this same interface.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

from BRAIN.llm.groq_client import LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    """
    Structural protocol for LLM backends.

    Both GroqClient and GeminiClient satisfy this without changes.
    brain.py types self._llm as LLMProvider for safety.
    """

    model: str
    temperature: float
    max_tokens: int

    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict],
    ) -> AsyncIterator[str]:
        ...

    async def call_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        max_tokens_override: Optional[int] = None,
    ) -> LLMResponse:
        ...

    async def stream_final(
        self,
        system_prompt: str,
        messages: List[Dict],
    ) -> AsyncIterator[str]:
        ...
