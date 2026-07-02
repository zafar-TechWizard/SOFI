"""
BRAIN/agents/runner.py — Production sub-agent runner.

Replaces orchestrator.py. This is the core agentic loop that sub-agents
execute in. Handles:
  - Iteration budget enforcement with grace call
  - Retry with jittered backoff (3 attempts)
  - Context overflow → compress messages
  - Content policy → stop with partial result
  - Interrupt check before each LLM call
  - Tool result truncation (15k chars)
  - Message history compression (80k chars total)
  - File write tracking for conflict detection
  - Progress callbacks via update_task_progress
  - Structured SubAgentResult assembly
  - Heartbeat activity reporting via registry
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from BRAIN.agents.budget import IterationBudget, TokenBudget
from BRAIN.agents.definitions import AGENT_DEFINITIONS
from BRAIN.agents.result import SubAgentResult, ToolTraceEntry, extract_summary
from BRAIN.agents.safety import check_tool_safety, filter_tools, BLOCKED_TOOLS
from BRAIN.llm.retry_utils import jittered_backoff

_log = logging.getLogger("sofi.brain.agents.runner")

MAX_TOOL_RESULT_CHARS = 15_000
MAX_TOTAL_MESSAGES_CHARS = 80_000
DEFAULT_MAX_ITERATIONS = 8
GRACE_ITERATIONS_LEFT = 1


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated from {len(text)} chars to {limit}]"


def _estimate_messages_chars(messages: List[Dict]) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(fn.get("arguments", ""))
    return total


def _compress_messages(messages: List[Dict], limit: int) -> List[Dict]:
    """
    Drop older tool-result pairs when total chars exceeds limit.
    Keeps the first message (user task brief) and the most recent messages.
    """
    if _estimate_messages_chars(messages) <= limit:
        return messages

    if len(messages) <= 3:
        return messages

    first = messages[0]
    remaining = messages[1:]

    while _estimate_messages_chars([first] + remaining) > limit and len(remaining) > 2:
        remaining.pop(0)

    return [first] + remaining


def _progress_tool_def(task_id: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "update_task_progress",
            "description": (
                "Report your progress on the current task. Call this after each "
                "major step to keep SOFi informed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step_description": {
                        "type": "string",
                        "description": "What you just completed",
                    },
                    "findings": {
                        "type": "string",
                        "description": "Key findings from this step",
                    },
                    "next_action": {
                        "type": "string",
                        "description": "What you're doing next",
                    },
                    "steps_plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Your full step plan (only on first call)",
                    },
                },
                "required": ["step_description", "findings"],
            },
        },
    }


def _handle_progress_update(args: dict, task_manager, task_id: str) -> str:
    if not task_manager or not task_id:
        return "Progress noted."

    steps_plan = args.get("steps_plan")
    if steps_plan:
        steps = [
            {"step": i + 1, "action": action, "status": "pending", "detail": ""}
            for i, action in enumerate(steps_plan)
        ]
        task_manager.set_steps(task_id, steps)

    task = task_manager.get_task(task_id)
    if task and task.steps:
        for i, step in enumerate(task.steps):
            if step.get("status") in ("pending", "in_progress"):
                detail = args.get("findings", "")
                task_manager.update_step(task_id, i, "done", detail)
                break

    next_action = args.get("next_action", "")
    if next_action:
        return f"Progress recorded. Continue with: {next_action}"
    return "Progress recorded."


class SubAgentRunner:
    """
    Production agentic loop for a single sub-agent.

    Lifecycle:
      1. Validate agent definition, filter tools
      2. Build system prompt (identity preamble + role instructions)
      3. Run LLM → tool → LLM loop with budget, retry, safety
      4. Assemble and return SubAgentResult
    """

    def __init__(
        self,
        agent_name: str,
        task_brief: str,
        tool_registry,
        llm,
        task_manager=None,
        task_id: str = "",
        subagent_id: str = "",
        max_iterations: int | None = None,
        timeout_seconds: float | None = None,
        registry=None,
        on_progress: Callable | None = None,
    ):
        self.agent_name = agent_name
        self.task_brief = task_brief
        self.tool_registry = tool_registry
        self.llm = llm
        self.task_manager = task_manager
        self.task_id = task_id
        self.subagent_id = subagent_id or f"sa-{uuid.uuid4().hex[:8]}"
        self.registry = registry
        self.on_progress = on_progress

        defn = AGENT_DEFINITIONS.get(agent_name)
        if defn is None:
            raise ValueError(
                f"Unknown agent '{agent_name}'. "
                f"Available: {', '.join(AGENT_DEFINITIONS.keys())}"
            )
        self._defn = defn

        max_iters = max_iterations or defn.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        self.budget = IterationBudget(max_iters)
        self.token_budget = TokenBudget(
            max_output_tokens=defn.get("max_output_tokens", 16_000),
        )

        self.timeout = timeout_seconds or defn.get("timeout_seconds", 300.0)
        self._files_written: Set[str] = set()
        self._tool_trace: List[ToolTraceEntry] = []
        self._start_time = 0.0
        self._output = ""
        self._error: str | None = None

    async def run(self) -> SubAgentResult:
        """Execute the sub-agent loop. Returns structured result."""
        self._start_time = time.time()

        system = self._defn["system"]
        allowed = set(filter_tools(self._defn["tools"])) - BLOCKED_TOOLS

        all_defs = self.tool_registry.get_definitions()
        agent_tool_defs = [
            d for d in all_defs
            if d["function"]["name"] in allowed
        ]

        if self.task_manager and self.task_id:
            agent_tool_defs.append(_progress_tool_def(self.task_id))
            self.task_manager.update_status(self.task_id, "in_progress")

        budget_hint = (
            f"\n\nBUDGET: You have {self.budget.max_iterations} iterations. "
            f"Use them wisely — do the most important work first."
        )

        messages: List[Dict] = [{"role": "user", "content": self.task_brief + budget_hint}]
        exit_reason = "max_iterations"

        try:
            exit_reason = await asyncio.wait_for(
                self._loop(system, messages, agent_tool_defs, allowed),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            exit_reason = "timeout"
            self._error = f"Agent timed out after {self.timeout:.0f}s"
            _log.warning(
                "runner | %s | timeout after %.0fs",
                self.subagent_id, self.timeout,
            )
        except asyncio.CancelledError:
            exit_reason = "interrupted"
            self._error = "Agent was interrupted"
        except Exception as exc:
            exit_reason = "error"
            self._error = str(exc)
            _log.exception("runner | %s | unhandled error", self.subagent_id)

        if self.task_manager and self.task_id:
            self.task_manager.update_status(self.task_id, "verifying")

        status = self._map_exit_to_status(exit_reason)
        duration = time.time() - self._start_time

        summary = extract_summary(self._output) if self._output else (self._error or "No output")

        return SubAgentResult(
            subagent_id=self.subagent_id,
            task_id=self.task_id,
            agent_type=self.agent_name,
            status=status,
            exit_reason=exit_reason,
            summary=summary,
            content=self._output,
            iterations=self.budget.used,
            duration_seconds=duration,
            tokens=self.token_budget.snapshot(),
            tool_trace=self._tool_trace,
            files_written=list(self._files_written),
            error=self._error,
        )

    async def _loop(
        self,
        system: str,
        messages: List[Dict],
        tool_defs: List[Dict],
        allowed: set,
    ) -> str:
        """Inner agentic loop. Returns exit_reason string."""

        while True:
            # ── Check interrupt ──
            if self.registry:
                rec = self.registry.get(self.subagent_id)
                if rec and rec.is_interrupted:
                    return "interrupted"

            # ── Check budget ──
            if self.budget.exhausted:
                if self.budget.remaining <= 0:
                    return "budget_exhausted"

            if not self.budget.consume():
                return "budget_exhausted"

            # ── Grace call: last iteration, tell the model to wrap up ──
            actual_tool_defs = tool_defs
            if self.budget.remaining <= GRACE_ITERATIONS_LEFT:
                grace_msg = {
                    "role": "system",
                    "content": (
                        "BUDGET ALERT: This is your LAST iteration. "
                        "Write your final complete output NOW. Do not call any more tools."
                    ),
                }
                messages.append(grace_msg)
                actual_tool_defs = []

            # ── Compress messages if too large ──
            messages = _compress_messages(messages, MAX_TOTAL_MESSAGES_CHARS)

            # ── Report heartbeat ──
            if self.registry:
                self.registry.touch(
                    self.subagent_id,
                    iteration=self.budget.used,
                )

            # ── LLM call with retry ──
            response = await self._call_llm_with_retry(system, messages, actual_tool_defs)
            if response is None:
                return "error"

            # ── Track tokens ──
            self.token_budget.record_chars(
                input_chars=_estimate_messages_chars(messages),
                output_chars=len(response.text or ""),
            )

            # ── Handle tool calls ──
            if response.tool_calls:
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                if response.raw_content is not None:
                    assistant_msg["_raw_content"] = response.raw_content
                messages.append(assistant_msg)

                results = await self._execute_tools(response.tool_calls, allowed)

                for tc, (tc_id, content) in zip(response.tool_calls, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": _truncate(content, MAX_TOOL_RESULT_CHARS),
                    })
                continue

            # ── Final text response ──
            if response.text:
                self._output = response.text
                _log.info(
                    "runner | %s | done after %d iter(s) | output_len=%d",
                    self.subagent_id, self.budget.used, len(self._output),
                )
                return "final_response"

            # ── Empty response — unusual but not fatal ──
            _log.warning("runner | %s | empty response at iter %d", self.subagent_id, self.budget.used)

        return "max_iterations"

    async def _call_llm_with_retry(self, system, messages, tool_defs, max_retries=3):
        """Call LLM with retry on transient errors. Returns None on unrecoverable failure."""
        last_exc = None

        for attempt in range(max_retries):
            try:
                return await self.llm.call_with_tools(system, messages, tool_defs)
            except Exception as exc:
                last_exc = exc
                classified = self.llm._classify_error(exc)
                _log.warning(
                    "runner | %s | LLM error attempt=%d cls=%s: %s",
                    self.subagent_id, attempt + 1, classified, exc,
                )

                if classified == "context_overflow":
                    messages[:] = _compress_messages(messages, MAX_TOTAL_MESSAGES_CHARS // 2)
                    _log.info("runner | %s | compressed messages for context overflow", self.subagent_id)
                    continue

                if classified == "content_policy":
                    self._error = f"Content policy: {exc}"
                    return None

                if classified in ("auth", "billing", "model_not_found", "format_error"):
                    self._error = f"Unrecoverable: {classified} — {exc}"
                    return None

                if classified in ("rate_limit", "server_error", "overloaded", "timeout"):
                    if attempt < max_retries - 1:
                        delay = jittered_backoff(attempt, base_delay=2.0, max_delay=30.0)
                        _log.info("runner | %s | retrying in %.1fs", self.subagent_id, delay)
                        await asyncio.sleep(delay)
                        continue

                if attempt == max_retries - 1:
                    self._error = f"LLM failed after {max_retries} attempts: {exc}"
                    return None

        self._error = f"LLM failed: {last_exc}"
        return None

    async def _execute_tools(self, tool_calls, allowed: set) -> List[tuple]:
        """Execute tool calls, returning list of (tc_id, result_string)."""
        from BRAIN.tools.registry import ToolCall as RegistryToolCall

        async def _exec_one(tc) -> tuple:
            start = time.time()

            # update_task_progress is internal
            if tc.name == "update_task_progress":
                result_str = _handle_progress_update(
                    tc.arguments, self.task_manager, self.task_id,
                )
                self.budget.refund()
                self._tool_trace.append(ToolTraceEntry(
                    tool=tc.name,
                    args_preview=tc.arguments.get("step_description", "")[:60],
                    result_bytes=len(result_str),
                    status="ok",
                    duration_ms=(time.time() - start) * 1000,
                ))
                return tc.id, result_str

            # Safety check
            tier, reason = check_tool_safety(tc.name, tc.arguments)
            if tier != "safe":
                _log.warning(
                    "runner | %s | tool blocked: %s — %s",
                    self.subagent_id, tc.name, reason,
                )
                self._tool_trace.append(ToolTraceEntry(
                    tool=tc.name,
                    args_preview=str(tc.arguments)[:60],
                    result_bytes=len(reason),
                    status="blocked",
                    duration_ms=(time.time() - start) * 1000,
                ))
                return tc.id, f"Tool blocked: {reason}"

            # Not in allowed set
            if tc.name not in allowed and tc.name != "update_task_progress":
                msg = f"Tool '{tc.name}' is not available to this agent."
                self._tool_trace.append(ToolTraceEntry(
                    tool=tc.name,
                    args_preview=str(tc.arguments)[:60],
                    result_bytes=len(msg),
                    status="blocked",
                    duration_ms=(time.time() - start) * 1000,
                ))
                return tc.id, msg

            # Report activity
            if self.registry:
                self.registry.touch(self.subagent_id, tool=tc.name)

            # Execute
            try:
                result = await self.tool_registry.execute(
                    RegistryToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                )
                result_str = result.to_string()
                status = "ok" if result.success else "error"

                # Track file writes
                if tc.name in ("write_file", "patch_file") and result.success:
                    path = tc.arguments.get("path", "") or tc.arguments.get("file_path", "")
                    if path:
                        self._files_written.add(path)

            except asyncio.TimeoutError:
                result_str = f"Tool '{tc.name}' timed out"
                status = "timeout"
            except Exception as exc:
                result_str = f"Tool error: {exc}"
                status = "error"

            elapsed = (time.time() - start) * 1000
            self._tool_trace.append(ToolTraceEntry(
                tool=tc.name,
                args_preview=str(tc.arguments)[:60],
                result_bytes=len(result_str),
                status=status,
                duration_ms=elapsed,
            ))

            # Clear tool from activity
            if self.registry:
                self.registry.touch(self.subagent_id, tool=None)

            return tc.id, result_str

        results = await asyncio.gather(
            *[_exec_one(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        final = []
        for tc, r in zip(tool_calls, results):
            if isinstance(r, Exception):
                final.append((tc.id, f"Tool execution error: {r}"))
            else:
                final.append(r)
        return final

    @staticmethod
    def _map_exit_to_status(exit_reason: str) -> str:
        return {
            "final_response": "completed",
            "max_iterations": "completed",
            "budget_exhausted": "completed",
            "timeout": "failed",
            "interrupted": "failed",
            "error": "failed",
        }.get(exit_reason, "failed")
