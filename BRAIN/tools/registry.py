"""
BRAIN/tools/registry.py — SOFi's tool registry

Central registry where all tools register themselves. Each tool has:
  - name + schema: what the LLM sees (OpenAI function-calling format)
  - handler: the async function that executes the tool
  - check_fn: returns True if the tool is available right now
  - needs_confirmation: True for sends/deletes (Claude Code permission model)
  - capability_name: maps to SelfModel for persona awareness

The ToolRegistry syncs with SelfModel — when a tool registers, SOFi's
prompt automatically reflects "I can do X" or "I can't do X right now."
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger("sofi.brain.tools")

from BRAIN.state.self_model import Capability, SelfModel


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_string(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error or 'unknown error'}"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

    @property
    def args_summary(self) -> str:
        parts = []
        for k, v in self.arguments.items():
            s = str(v)
            if len(s) > 40:
                s = s[:37] + "..."
            parts.append(f"{k}={s}")
        return ", ".join(parts) if parts else "(no args)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class ToolEntry:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable
    check_fn: Optional[Callable] = None
    needs_confirmation: bool = False
    # background=True → fire-and-forget dispatch; SOFi does NOT block on result.
    # Result flows back via AgenticWorkspace and surfaces next turn.
    background: bool = False
    timeout: float = 60.0
    category: str = "general"
    capability_name: Optional[str] = None
    capability_description: Optional[str] = None
    capability_refusal: Optional[str] = None

    def is_available(self) -> bool:
        if self.check_fn is None:
            return True
        try:
            return bool(self.check_fn())
        except Exception:
            return False


class ToolRegistry:

    def __init__(self) -> None:
        self._tools: Dict[str, ToolEntry] = {}
        self._self_model: Optional[SelfModel] = None

    def set_self_model(self, self_model: SelfModel) -> None:
        """Wire SelfModel so execute() can pre-check capability state."""
        self._self_model = self_model

    def register(self, entry: ToolEntry) -> None:
        if not entry.name:
            raise ValueError("ToolEntry.name must be non-empty")
        self._tools[entry.name] = entry

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolEntry]:
        return self._tools.get(name)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_available_tools(self) -> List[ToolEntry]:
        return [t for t in self._tools.values() if t.is_available()]

    def get_definitions(self) -> List[Dict[str, Any]]:
        defs = []
        for tool in self._tools.values():
            if not tool.is_available():
                continue
            defs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            })
        return defs

    def needs_confirmation(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool is None:
            return True
        return tool.needs_confirmation

    def is_background(self, name: str) -> bool:
        """True if the tool should be dispatched fire-and-forget (non-blocking)."""
        tool = self._tools.get(name)
        return tool.background if tool else False

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_call.name}",
            )
        if not tool.is_available():
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_call.name}' is not available right now.",
            )

        # Self-model gate: if capability was explicitly marked offline/unbuilt,
        # return honest refusal without executing — prevents wasted LLM retries.
        if self._self_model is not None:
            cap_name = tool.capability_name or tool.name
            reason = self._self_model.get_unavailable_reason(cap_name)
            if reason:
                _log.info(
                    "execute | self-model gate | tool=%s reason=%s",
                    tool_call.name, reason,
                )
                return ToolResult(
                    success=False,
                    output="",
                    error=reason,
                )

        from BRAIN.observability.metrics import get_metrics
        _m = get_metrics()

        t0 = time.perf_counter()
        try:
            result = tool.handler(**tool_call.arguments)
            if hasattr(result, "__await__"):
                result = await asyncio.wait_for(result, timeout=tool.timeout)
            duration_ms = (time.perf_counter() - t0) * 1000

            _m.inc("tool_executions")
            _m.observe("tool_duration_ms", duration_ms)

            output = str(result) if result is not None else "Done."
            from BRAIN.tools.output_limits import truncate_result
            output = truncate_result(tool_call.name, output)
            return ToolResult(
                success=True,
                output=output,
                duration_ms=duration_ms,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - t0) * 1000
            _m.inc("tool_errors")
            _m.inc("tool_timeouts")
            _m.observe("tool_duration_ms", duration_ms)
            _log.warning(
                "execute | timeout | tool=%s after %.0fms (limit=%.0fs)",
                tool_call.name, duration_ms, tool.timeout,
            )
            return ToolResult(
                success=False,
                output="",
                error=f"Tool timed out after {tool.timeout:.0f}s.",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            _m.inc("tool_errors")
            _m.observe("tool_duration_ms", duration_ms)
            return ToolResult(
                success=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
            )

    def sync_with_self_model(self, self_model: SelfModel) -> None:
        for tool in self._tools.values():
            cap_name = tool.capability_name or tool.name
            cap_desc = tool.capability_description or tool.description
            cap_refusal = tool.capability_refusal or f"I can't use {tool.name} right now."
            available = tool.is_available()

            self_model.register(Capability(
                name=cap_name,
                description=cap_desc,
                refusal_offline=cap_refusal,
                refusal_not_built="",
                installed=True,
                available=available,
            ))

    def status(self) -> Dict[str, Any]:
        available = self.get_available_tools()
        return {
            "registered": len(self._tools),
            "available": len(available),
            "tools": {
                name: {
                    "available": tool.is_available(),
                    "category": tool.category,
                    "needs_confirmation": tool.needs_confirmation,
                    "background": tool.background,
                }
                for name, tool in self._tools.items()
            },
        }
