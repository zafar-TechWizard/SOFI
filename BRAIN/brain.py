"""
BRAIN/brain.py — SOFi Coordinator

The agentic loop. Each turn:
  1. observe(user, message)         — feeds memory
  2. get_context_async()            — pulls back must_know / context / assoc / turns
  3. infer user state + decide mode — non-LLM, rule-based
  4. build prompt + messages         — persona block + memory sections + recent turns
  5. AGENTIC LOOP:
       call LLM with tools →
       if tool_calls: execute (inline OR background), append results, continue
       if text only: stream to user, break
  6. observe(assistant, response)   — captures the answer

Background tool pattern:
  - Tools marked background=True are dispatched fire-and-forget
  - SOFi acknowledges immediately and continues the conversation
  - The tool runs concurrently; result lands in AgenticWorkspace
  - Next turn: WHAT I'VE BEEN DOING section surfaces the result naturally

Public API:
    brain = Brain()
    await brain.setup(on_progress=optional_callback)
    async for token in brain.process(user_message):
        ...
    brain.clear_history()
    await brain.shutdown()
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

from memory.memory_manager import MemoryManager
from memory.working_memory.working_context import (
    NotifyPriority,
    WorkspaceItem,
    WorkspaceItemStatus,
    WorkspaceItemType,
)

import os

from BRAIN.llm import GeminiClient, GroqClient
from BRAIN.mode import Mode, ModeController
from BRAIN.persona.persona import (
    DEFAULT_MODE,
    get_personality_dict,
    set_self_model,
    warm_cache,
)
from BRAIN.prompt import build_messages, build_prompt
from BRAIN.state import SelfModel, UserStateInferencer, UserStateUpdate
from BRAIN.tools.registry import ToolCall, ToolRegistry, ToolResult


_log = logging.getLogger("sofi.brain")

ProgressFn = Callable[[str], None]
ToolEventFn = Optional[Callable[[str, Dict[str, Any]], None]]


def _time_ago(dt: datetime) -> str:
    """Human-readable elapsed time for log/display."""
    delta = (datetime.now() - dt).total_seconds()
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    return f"{int(delta / 3600)}h ago"


class Brain:
    """
    Single coordinator for SOFi.

    Owns MemoryManager, GroqClient, ToolRegistry, SelfModel. Each
    `process()` call drives the full agentic loop.

    Inline tools (background=False): await result, feed back to LLM immediately.
    Background tools (background=True): fire-and-forget, result via AgenticWorkspace.
    """

    MAX_LOCAL_TURNS = 20
    MAX_TOOL_ITERATIONS = 10

    def __init__(
        self,
        model: Optional[str] = None,
        mode: str = DEFAULT_MODE,
        memory_log: bool = False,
        memory_review: bool = False,
    ):
        self._llm: Optional[Any] = None
        self._memory: Optional[MemoryManager] = None
        self._model_override = model
        self._mode: Mode = Mode(mode) if isinstance(mode, str) else mode
        self._prev_mode: Mode = self._mode
        self._prev_was_override: bool = False
        self._forced_mode: Optional[Mode] = None
        self._mem_log = memory_log
        self._mem_review = memory_review
        self._is_ready: bool = False
        self._local_history: list[dict] = []
        self._user_state_inferencer = UserStateInferencer()
        self._mode_controller = ModeController()
        self._prev_user_state: Optional[UserStateUpdate] = None
        self._last_mode_decision: Optional[dict] = None
        self._last_user_state: Optional[dict] = None
        self._self_model: Optional[SelfModel] = None
        self._tool_registry: ToolRegistry = ToolRegistry()
        self._last_tool_calls: List[Dict[str, Any]] = []
        self._on_tool_event: ToolEventFn = None

        # Background task tracking — keeps asyncio tasks alive until done
        self._background_tasks: Set[asyncio.Task] = set()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def setup(self, on_progress: Optional[ProgressFn] = None) -> None:
        """Full cold-start setup. ~30 min first run (Docker + GLiNER + Neo4j)."""
        if self._is_ready:
            return
        await self._setup_core(on_progress)
        await self._setup_memory(on_progress)

    async def _setup_core(self, on_progress: Optional[ProgressFn] = None) -> None:
        """
        Fast path — persona + LLM + tools. ~2-3 seconds.
        No Docker, no Neo4j, no ML models. Safe to call on every hot reload.
        """
        def _tick(stage: str) -> None:
            _log.info("setup | %s", stage)
            if on_progress:
                try:
                    on_progress(stage)
                except Exception:
                    pass

        _tick("loading persona")
        self._self_model = SelfModel.from_personality(get_personality_dict())
        set_self_model(self._self_model)
        warm_cache()

        # LLM backend: SOFI_LLM_BACKEND=groq to switch; default is Gemini.
        backend = os.environ.get("SOFI_LLM_BACKEND", "gemini").lower().strip()
        if backend == "groq":
            _tick("connecting Groq")
            self._llm = GroqClient(model=self._model_override)
        else:
            _tick("connecting Gemini")
            self._llm = GeminiClient(model=self._model_override)

        _tick("registering tools")
        self._tool_registry = ToolRegistry()
        _auto_register_tools(self._tool_registry)
        from BRAIN.tools._agent_tools import register_agent_tools
        register_agent_tools(self._tool_registry, self._llm)
        self._tool_registry.sync_with_self_model(self._self_model)
        set_self_model(self._self_model)
        warm_cache()

        tool_status = self._tool_registry.status()
        _log.info(
            "setup | tools ready | registered=%d available=%d "
            "background=%d inline=%d",
            tool_status["registered"],
            tool_status["available"],
            sum(1 for t in self._tool_registry._tools.values() if t.background),
            sum(1 for t in self._tool_registry._tools.values() if not t.background),
        )

    async def _setup_memory(self, on_progress: Optional[ProgressFn] = None) -> None:
        """
        Slow path — Docker + Neo4j + GLiNER + cross-encoder. ~10-30 min cold.
        Only called once at full startup. Skipped on hot reload.
        """
        def _tick(stage: str) -> None:
            _log.info("setup | %s", stage)
            if on_progress:
                try:
                    on_progress(stage)
                except Exception:
                    pass

        _tick("starting memory (Docker + Neo4j + models)…")
        self._memory = MemoryManager(log=self._mem_log, review=self._mem_review)
        await self._memory.setup()

        _tick("ready")
        self._is_ready = True

    async def hot_reload(self) -> "Brain":
        """
        Reload all BRAIN source modules and return a fresh Brain instance that
        reuses the existing MemoryManager (Neo4j + GLiNER + cross-encoder stay live).

        Called by /reload in the CLI. Total cost: ~2-3 seconds.
        The caller must swap its reference from the old brain to the returned one.
        """
        import importlib
        import sys

        _log.info("hot_reload | starting")

        # Reload in dependency order: leaves first, brain last.
        RELOAD_ORDER = [
            "BRAIN.persona.persona",
            "BRAIN.state.user_state",
            "BRAIN.state.self_model",
            "BRAIN.state",
            "BRAIN.mode.signals",
            "BRAIN.mode.controller",
            "BRAIN.mode",
            "BRAIN.prompt.formatters",
            "BRAIN.prompt.builder",
            "BRAIN.prompt",
            "BRAIN.llm.groq_client",
            "BRAIN.llm.gemini_client",
            "BRAIN.llm",
            "BRAIN.tools.registry",
            "BRAIN.tools",
            "BRAIN.agents.definitions",
            "BRAIN.agents.orchestrator",
            "BRAIN.agents",
            "BRAIN.tools._agent_tools",
            "BRAIN.skills._registry",
            "BRAIN.brain",
        ]

        # Also dynamically include all tool modules found in BRAIN/tools/
        # so newly added tool files are picked up on hot-reload without
        # needing a manual RELOAD_ORDER edit.
        from pathlib import Path as _HRPath
        _tools_dir = _HRPath(__file__).parent / "tools"
        for _p in sorted(_tools_dir.glob("*.py")):
            if not _p.stem.startswith("_"):
                _mod = f"BRAIN.tools.{_p.stem}"
                if _mod not in RELOAD_ORDER:
                    RELOAD_ORDER.insert(RELOAD_ORDER.index("BRAIN.tools"), _mod)

        reloaded, failed = 0, 0
        for mod_name in RELOAD_ORDER:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reloaded += 1
                    _log.debug("hot_reload | ok  %s", mod_name)
                except Exception as exc:
                    failed += 1
                    _log.warning("hot_reload | ERR %s: %s", mod_name, exc)

        _log.info("hot_reload | modules reloaded=%d failed=%d", reloaded, failed)

        # Import fresh Brain class from the just-reloaded module.
        from BRAIN.brain import Brain as FreshBrain

        new_brain = FreshBrain(
            model=self._model_override,
            memory_log=self._mem_log,
            memory_review=self._mem_review,
        )

        # Inject the existing memory stack — the only thing we can't reload cheaply.
        new_brain._memory = self._memory

        # Carry over in-session conversational state so the reload is invisible to SOFi.
        new_brain._local_history = list(self._local_history)
        new_brain._prev_user_state = self._prev_user_state
        new_brain._prev_mode = self._prev_mode
        new_brain._prev_was_override = self._prev_was_override
        new_brain._forced_mode = self._forced_mode

        # Run the fast path only — persona + LLM + tools (~2-3s, no Docker).
        await new_brain._setup_core()
        new_brain._is_ready = True

        _log.info(
            "hot_reload | complete | tools=%d",
            new_brain._tool_registry.tool_count,
        )
        return new_brain

    async def shutdown(self) -> None:
        # Wait briefly for any in-flight background tasks
        if self._background_tasks:
            _log.info("shutdown | waiting for %d background tasks", len(self._background_tasks))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                _log.warning("shutdown | background tasks timed out — cancelling")
                for t in self._background_tasks:
                    t.cancel()

        # Auto-run consolidation in background on shutdown so memories accumulate
        # without Zafar having to remember to run it manually.
        # Runs in a daemon thread so it doesn't block the shutdown sequence.
        _trigger_consolidation_on_shutdown()

        if self._memory:
            try:
                await self._memory.shutdown(stop_docker=False)
            except Exception:
                pass
        self._is_ready = False

    # =========================================================================
    # Conversation — The Agentic Loop
    # =========================================================================

    async def process(self, message: str) -> AsyncIterator[str]:
        self._require_ready()
        self._last_tool_calls = []

        self._local_history.append({"role": "user", "content": message})
        self._local_history = self._local_history[-self.MAX_LOCAL_TURNS:]

        # ─── PHASE A: Context gathering (unchanged) ───
        await self._memory.observe("user", message)
        await self._memory.get_context_async("user", message)
        ctx = self._memory.get_full_context()

        # ─── PHASE B: State + Mode inference (unchanged) ───
        user_state = self._user_state_inferencer.infer(
            ctx, message, prev_state=self._prev_user_state,
        )
        try:
            self._memory.context_manager.update_user_state(**user_state.as_dict())
            ctx = self._memory.get_full_context()
        except Exception:
            pass
        self._prev_user_state = user_state
        self._last_user_state = user_state.as_dict()

        if self._forced_mode is not None:
            from BRAIN.mode.controller import ModeDecision
            decision = ModeDecision(
                mode=self._forced_mode,
                allow_dropped_formality=(
                    self._forced_mode == Mode.EMPATHETIC
                    and (user_state.emotional_intensity or 0.0) >= 0.60
                ),
                scores={},
                triggered_overrides=["forced_by_user"],
                held_prev=False,
            )
        else:
            decision = self._mode_controller.decide(
                ctx, message,
                prev_mode=self._prev_mode,
                prev_was_override=self._prev_was_override,
            )
        self._mode = decision.mode
        self._prev_mode = decision.mode
        self._prev_was_override = any(
            t in ("intensity_override", "explicit_creative_phrase", "code_block_present")
            for t in decision.triggered_overrides
        )
        self._last_mode_decision = decision.as_dict()

        # ─── PHASE C: Build prompt + tool definitions ───
        action_state = self._get_action_state()

        _log.debug(
            "process | mode=%s emotion=%s intensity=%.2f action_state_keys=%s",
            decision.mode.value,
            user_state.current_emotional_state,
            user_state.emotional_intensity or 0.0,
            list(action_state.keys()) if action_state else [],
        )

        system_prompt = build_prompt(
            ctx,
            mode=decision.mode.value,
            allow_dropped_formality=decision.allow_dropped_formality,
            action_state=action_state,
        )

        messages = build_messages(ctx, message)
        memory_recent = getattr(getattr(ctx, "memory", None), "recent_turns", None)
        if not memory_recent:
            messages = list(self._local_history)

        tool_defs = self._tool_registry.get_definitions()

        # ─── PHASE D: THE AGENTIC LOOP ───
        # Inline tools: await result → LLM sees it immediately → continues.
        # Background tools: fire-and-forget → placeholder result → LLM acknowledges → moves on.
        #
        # Two separate accumulators:
        #   display_text — everything yielded to the CLI (includes tool markers)
        #   response_text — only LLM-generated text (saved to memory)

        display_text = ""
        response_text = ""
        iteration = 0

        if not tool_defs:
            # No tools registered — pure conversation, stream directly.
            _log.debug("process | no tools registered — pure conversation stream")
            async for token in self._llm.stream(system_prompt, messages):
                response_text += token
                yield token
        else:
            _log.debug("process | agentic loop start | tools=%d", len(tool_defs))

            while iteration < self.MAX_TOOL_ITERATIONS:
                iteration += 1
                _log.debug("process | agentic loop iter=%d", iteration)

                try:
                    response = await self._llm.call_with_tools(
                        system_prompt, messages, tool_defs,
                    )
                except Exception as exc:
                    error_class = self._llm._classify_error(exc)
                    _log.warning(
                        "process | LLM error | class=%s | exc=%s",
                        error_class, exc, exc_info=True,
                    )
                    # Always surface errors — silent failures are worse than noisy ones.
                    if error_class == "rate_limit":
                        error_msg = (
                            "\n\nI've hit a rate limit. Give it a moment and try again, sir."
                        )
                    elif error_class == "auth":
                        error_msg = "\n\nAPI key issue — check gemini_api_key in .env."
                    elif error_class == "content_filter":
                        error_msg = "\n\nMy response was blocked by a content filter. Try rephrasing."
                    else:
                        error_msg = (
                            f"\n\nSomething went wrong on my end ({error_class}): {exc}"
                        )
                    response_text += error_msg
                    yield error_msg
                    break

                # ── CASE: finish_reason == "length" ──
                if response.finish_reason == "length":
                    _log.debug("process | finish_reason=length — continuing")
                    if response.text:
                        response_text += response.text
                        yield response.text
                    messages.append({"role": "assistant", "content": response.text or ""})
                    messages.append({"role": "user", "content": "Please continue your response."})
                    continue

                # ── CASE: tool_calls present ──
                if response.tool_calls:
                    # Suppress response.text here — when a model makes tool calls it
                    # often emits chain-of-thought reasoning as the text part ("The user
                    # wants to... I should..."). Showing that breaks persona and feels like
                    # a chatbot narrating itself. Tool labels ("`reading emails...`") already
                    # tell the user what's happening. The clean final response comes after
                    # inline tools execute and we loop back.
                    if response.text:
                        _log.debug(
                            "process | suppressing LLM reasoning text alongside tools | "
                            "len=%d preview=%.80s", len(response.text), response.text,
                        )

                    # ── Acknowledgement-first for slow/multi-tool batches ──
                    # Fast tools (read_file, list_directory, get_current_time) run
                    # silently — they're quick enough that an ack would just be noise.
                    # Slow tools (web_search, run_command, write_file, etc.) or batches
                    # of 2+ tools get a brief ack so Zafar knows she's on it immediately,
                    # before the work starts.
                    ack_text = _ack_for_tools(response.tool_calls)
                    if ack_text and iteration == 1:
                        # Only ack on the FIRST iteration — subsequent tool rounds
                        # are follow-up actions inside the same turn; no need to ack again.
                        display_text += ack_text
                        response_text += ack_text
                        yield ack_text

                    # Build assistant message with tool_calls (Groq/OpenAI format).
                    # For Gemini thinking models, also carry raw_content so that
                    # _messages_to_contents can pass the original Content object back
                    # to the API verbatim — preserving the thought_signature bytes
                    # inside FunctionCall parts that the 400 error requires.
                    assistant_msg: Dict[str, Any] = {
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

                    # ── Phase 1: Show all labels upfront + dispatch background tools ──
                    # All tool labels appear immediately so Zafar sees the full
                    # work plan before any tool has finished executing.
                    had_inline = False
                    inline_tcs = []
                    bg_dispatched: Dict[str, Any] = {}  # tc.id → workspace_item_id

                    for tc in response.tool_calls:
                        is_bg = self._tool_registry.is_background(tc.name)

                        tool_label = _tool_display_name(tc.name, is_bg)
                        status_line = f"\n\n`{tool_label}`\n\n"
                        display_text += status_line
                        yield status_line

                        self._emit_tool_event("tool_start", {
                            "name": tc.name,
                            "args": tc.arguments,
                            "iteration": iteration,
                            "background": is_bg,
                        })

                        if is_bg:
                            _log.info(
                                "process | background dispatch | tool=%s args=%s",
                                tc.name, _args_preview(tc.arguments),
                            )
                            item_id = self._dispatch_background(
                                ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                            )
                            bg_dispatched[tc.id] = item_id
                            self._emit_tool_event("tool_dispatched", {
                                "name": tc.name,
                                "args": tc.arguments,
                                "iteration": iteration,
                                "workspace_item_id": item_id,
                            })
                        else:
                            had_inline = True
                            inline_tcs.append(tc)

                    # ── Phase 2: Execute all inline tools in parallel ──
                    # asyncio.gather runs them concurrently; wall-clock time =
                    # slowest single tool instead of sum of all tools.
                    # return_exceptions=True prevents one failure from cancelling others.
                    inline_results: Dict[str, Any] = {}  # tc.id → result | Exception

                    if inline_tcs:
                        async def _exec_tc(tc):
                            _log.debug("process | inline execute | tool=%s", tc.name)
                            return await self._tool_registry.execute(
                                ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                            )

                        gathered = await asyncio.gather(
                            *[_exec_tc(tc) for tc in inline_tcs],
                            return_exceptions=True,
                        )

                        for tc, result in zip(inline_tcs, gathered):
                            inline_results[tc.id] = result
                            if isinstance(result, Exception):
                                _log.error(
                                    "process | inline tool error | tool=%s exc=%s",
                                    tc.name, result,
                                )
                                self._emit_tool_event("tool_end", {
                                    "name": tc.name,
                                    "success": False,
                                    "duration_ms": 0,
                                    "iteration": iteration,
                                })
                            else:
                                _log.info(
                                    "process | inline complete | tool=%s success=%s "
                                    "duration_ms=%.0f output=%.80s",
                                    tc.name, result.success, result.duration_ms,
                                    result.output[:80] if result.output else "",
                                )
                                self._emit_tool_event("tool_end", {
                                    "name": tc.name,
                                    "success": result.success,
                                    "duration_ms": result.duration_ms,
                                    "iteration": iteration,
                                })
                                self._last_tool_calls.append({
                                    "name": tc.name,
                                    "args": tc.arguments,
                                    "success": result.success,
                                    "duration_ms": result.duration_ms,
                                    "output_preview": result.output[:200] if result.output else "",
                                })

                    # ── Phase 3: Append tool messages in original tool_calls order ──
                    # Preserving order keeps Groq and Gemini message history valid.
                    for tc in response.tool_calls:
                        if tc.id in bg_dispatched:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "Dispatched to background — running independently.",
                            })
                        elif tc.id in inline_results:
                            result = inline_results[tc.id]
                            if isinstance(result, Exception):
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": f"Tool error: {result}",
                                })
                            else:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": result.to_string(),
                                })

                    # ── Decide whether to loop back ──
                    if had_inline:
                        # Inline tools ran — LLM needs to see their results.
                        # Loop back for the follow-up response.
                        _log.debug("process | had inline tools — continuing loop")
                        continue
                    else:
                        # All tools were background (fire-and-forget).
                        # We suppressed response.text above (it was reasoning), so
                        # always emit a brief acknowledgement so the turn isn't silent.
                        ack = "On it."
                        response_text += ack
                        yield ack
                        _log.debug(
                            "process | all-background turn — ending cleanly | "
                            "response_text_len=%d", len(response_text),
                        )
                        break

                # ── CASE: text only (final response) ──
                if response.text:
                    _log.debug(
                        "process | final text | len=%d preview=%.80s",
                        len(response.text), response.text,
                    )
                    response_text += response.text
                    yield response.text
                break

            # Safety: max iterations hit with no text response
            if iteration >= self.MAX_TOOL_ITERATIONS and not response_text:
                _log.warning("process | max_iterations=%d hit without text response", self.MAX_TOOL_ITERATIONS)
                fallback = "I seem to have gotten caught in a loop, sir. Let me answer directly."
                response_text = fallback
                yield fallback

        # ─── PHASE E: Post-response ───
        # Save only the LLM-generated text to memory (not tool status markers)
        if response_text:
            await self._memory.observe("assistant", response_text)
            self._local_history.append({"role": "assistant", "content": response_text})
            self._local_history = self._local_history[-self.MAX_LOCAL_TURNS:]
            _log.debug("process | response saved to memory | len=%d", len(response_text))

    def clear_history(self) -> None:
        self._local_history.clear()
        self._last_tool_calls.clear()

    # =========================================================================
    # Background tool execution
    # =========================================================================

    def _dispatch_background(self, tool_call: ToolCall) -> str:
        """
        Fire-and-forget dispatch of a background tool.
        Creates an IN_PROGRESS WorkspaceItem immediately, then schedules
        _run_background() as an asyncio task (non-blocking).
        Returns the workspace item id for tracking.
        """
        ws = self._get_workspace()
        item_id = ""

        if ws is not None:
            item = WorkspaceItem(
                type=WorkspaceItemType.TASK,
                title=tool_call.name,
                description=f"Args: {tool_call.args_summary}",
                status=WorkspaceItemStatus.IN_PROGRESS,
                progress=0.0,
                notify=False,   # Will be set True on completion
                source_agent="background_executor",
                metadata={
                    "tool": tool_call.name,
                    "args": tool_call.arguments,
                    "dispatched_at": datetime.now().isoformat(),
                },
            )
            item_id = ws.add_item(item)
            _log.info(
                "background | workspace item created | id=%s tool=%s",
                item_id, tool_call.name,
            )
        else:
            _log.warning("background | no workspace available | tool=%s", tool_call.name)

        # Schedule the background coroutine — fire-and-forget
        task = asyncio.create_task(
            self._run_background(tool_call, item_id),
            name=f"bg-{tool_call.name}-{item_id[:8]}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        _log.debug(
            "background | task scheduled | tool=%s item_id=%s active_bg_tasks=%d",
            tool_call.name, item_id, len(self._background_tasks),
        )
        return item_id

    async def _run_background(self, tool_call: ToolCall, workspace_item_id: str) -> None:
        """
        Execute a background tool and write the result to AgenticWorkspace.
        SOFi's main turn is NOT blocked on this — it runs concurrently.
        """
        t0 = time.perf_counter()
        _log.info("background | starting | tool=%s id=%s", tool_call.name, workspace_item_id[:8])

        try:
            result = await self._tool_registry.execute(tool_call)
            duration_ms = (time.perf_counter() - t0) * 1000

            ws = self._get_workspace()
            if ws and workspace_item_id:
                new_status = (
                    WorkspaceItemStatus.COMPLETED if result.success
                    else WorkspaceItemStatus.FAILED
                )
                ws.update_item(
                    workspace_item_id,
                    status=new_status,
                    progress=1.0 if result.success else 0.0,
                    notify=True,                        # surface to SOFi next turn
                    notify_priority=NotifyPriority.LOW, # queue for next user turn
                    description=(
                        result.output[:300] if result.success
                        else f"Error: {result.error}"
                    ),
                    metadata={
                        "tool": tool_call.name,
                        "args": tool_call.arguments,
                        "output": result.output,
                        "success": result.success,
                        "duration_ms": duration_ms,
                        "completed_at": datetime.now().isoformat(),
                    },
                )

            if result.success:
                _log.info(
                    "background | completed | tool=%s duration_ms=%.0f "
                    "output=%.120s",
                    tool_call.name, duration_ms,
                    result.output[:120] if result.output else "",
                )
            else:
                _log.warning(
                    "background | failed | tool=%s duration_ms=%.0f error=%s",
                    tool_call.name, duration_ms, result.error,
                )

        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            _log.error(
                "background | exception | tool=%s duration_ms=%.0f error=%s",
                tool_call.name, duration_ms, exc, exc_info=True,
            )
            ws = self._get_workspace()
            if ws and workspace_item_id:
                ws.update_item(
                    workspace_item_id,
                    status=WorkspaceItemStatus.FAILED,
                    progress=0.0,
                    notify=True,
                    description=f"Exception: {exc}",
                    metadata={
                        "tool": tool_call.name,
                        "error": str(exc),
                        "completed_at": datetime.now().isoformat(),
                    },
                )

    def _get_workspace(self):
        """Safe accessor for AgenticWorkspace. Returns None if memory not ready."""
        if not self._memory:
            return None
        try:
            ctx = self._memory.get_full_context()
            return getattr(ctx, "workspace", None)
        except Exception as exc:
            _log.warning("_get_workspace | error | %s", exc)
            return None

    # =========================================================================
    # Tool event callback — wired by CLI for live display
    # =========================================================================

    def set_tool_event_handler(self, handler: ToolEventFn) -> None:
        self._on_tool_event = handler

    def _emit_tool_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._on_tool_event:
            try:
                self._on_tool_event(event_type, data)
            except Exception:
                pass

    # =========================================================================
    # Introspection
    # =========================================================================

    @property
    def mode(self) -> str:
        return self._mode.value if isinstance(self._mode, Mode) else str(self._mode)

    def set_mode(self, mode: str) -> None:
        self._mode = Mode(mode)
        self._prev_mode = self._mode

    @property
    def last_mode_decision(self) -> Optional[dict]:
        return self._last_mode_decision

    @property
    def last_user_state(self) -> Optional[dict]:
        return self._last_user_state

    @property
    def last_tool_calls(self) -> List[Dict[str, Any]]:
        return list(self._last_tool_calls)

    def force_mode(self, mode_name: str) -> None:
        if mode_name == "auto":
            self._forced_mode = None
            return
        self._forced_mode = Mode(mode_name)
        self._mode = self._forced_mode
        self._prev_mode = self._forced_mode

    def tool_status(self) -> dict:
        return self._tool_registry.status()

    def inspect(self) -> dict:
        snap = {
            "mode": self.mode,
            "mode_decision": self._last_mode_decision,
            "user_state": self._last_user_state,
            "self_model": self._self_model.snapshot() if self._self_model else None,
            "tools": self._tool_registry.status(),
            "last_tool_calls": self._last_tool_calls,
            "background_tasks_active": len(self._background_tasks),
        }

        ws = self._get_workspace()
        if ws:
            snap["workspace"] = ws.snapshot()

        if self._memory is not None:
            try:
                ctx = self._memory.get_full_context()
            except Exception:
                ctx = None
            if ctx is not None:
                mem = getattr(ctx, "memory", None)
                sofi = getattr(ctx, "sofi", None)
                if mem is not None:
                    snap["memory"] = {
                        "must_know":    list(mem.must_know or []),
                        "context":      list(mem.context or []),
                        "associations": list(mem.associations or []),
                    }
                if sofi is not None:
                    snap["sofi"] = {
                        "current_datetime": getattr(sofi, "current_datetime", None),
                        "time_of_day":      getattr(sofi, "time_of_day", None),
                    }
        return snap

    @property
    def memory(self) -> Optional[MemoryManager]:
        return self._memory

    @property
    def self_model(self) -> Optional[SelfModel]:
        return self._self_model

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    def register_capability(self, capability) -> None:
        if self._self_model is None:
            raise RuntimeError(
                "SelfModel not initialised — call await brain.setup() first."
            )
        self._self_model.register(capability)
        set_self_model(self._self_model)

    # =========================================================================
    # Internals
    # =========================================================================

    def _get_action_state(self) -> Optional[Dict[str, Any]]:
        """
        Assemble the action state dict for prompt injection.

        Reads from:
          - self._last_tool_calls — inline tools that ran last turn
          - AgenticWorkspace     — background tasks (in-progress + just completed)

        Notifications are cleared (notify=False) after being surfaced so they
        appear exactly once — SOFi sees and acknowledges them this turn only.
        """
        state: Dict[str, Any] = {}

        # ── Inline tool completions from last turn ──
        if self._last_tool_calls:
            completed = [
                {
                    "summary": f"{tc['name']}({_args_preview(tc.get('args', {}))})",
                    "ago": f"{tc.get('duration_ms', 0):.0f}ms",
                }
                for tc in self._last_tool_calls[-3:]
            ]
            if completed:
                state["completed"] = completed

        # ── AgenticWorkspace: background tasks ──
        ws = self._get_workspace()
        if ws:
            # Active background tasks (still running)
            active_tasks = ws.get_active_tasks()
            if active_tasks:
                state["active"] = [
                    {
                        "title": item.title,
                        "ago": _time_ago(item.created_at),
                    }
                    for item in active_tasks[:3]
                ]
                _log.debug(
                    "_get_action_state | active background tasks: %s",
                    [t.title for t in active_tasks],
                )

            # Background tasks that just completed — surface to SOFi this turn
            notifications = ws.get_pending_notifications()
            if notifications:
                state["notifications"] = [
                    {
                        "summary": (
                            f"{item.title} — "
                            f"{'done' if item.status == WorkspaceItemStatus.COMPLETED else 'failed'}: "
                            f"{item.description[:150]}"
                        ),
                    }
                    for item in notifications[:3]
                ]
                _log.info(
                    "_get_action_state | surfacing %d background notifications: %s",
                    len(notifications),
                    [n.title for n in notifications],
                )

                # Clear notifications so they don't repeat next turn
                for item in notifications:
                    ws.update_item(item.id, notify=False)
                    _log.debug(
                        "_get_action_state | notification cleared | id=%s title=%s",
                        item.id[:8], item.title,
                    )

        if not state:
            return None

        _log.debug("_get_action_state | state_keys=%s", list(state.keys()))
        return state

    def _require_ready(self) -> None:
        if not self._is_ready or self._llm is None or self._memory is None:
            raise RuntimeError(
                "Brain is not initialized. Call `await brain.setup()` first."
            )


# =============================================================================
# Helpers
# =============================================================================

def _trigger_consolidation_on_shutdown() -> None:
    """
    Fire consolidation in a background daemon thread on shutdown.

    Only runs if there are conversation logs newer than the last consolidation
    run. Non-blocking — returns immediately. Daemon thread dies with the process
    if it hasn't finished, so it never delays an exit.
    """
    import threading
    from pathlib import Path as _P

    def _run():
        try:
            # Quick check: any conversation log files exist?
            log_dir = _P(__file__).parent / "memory" / "data"
            conv_file = log_dir / "conversation.json"
            if not conv_file.exists():
                _log.debug("consolidation | no conversation log — skipping")
                return

            _log.info("consolidation | auto-running on shutdown")
            import subprocess, sys as _sys
            proc = subprocess.Popen(
                [_sys.executable, "-m", "memory.processing.consolidation_runner"],
                cwd=str(_P(__file__).parent.parent),  # workspace root
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            proc.wait(timeout=400)
            _log.info("consolidation | completed (exit=%d)", proc.returncode)
        except Exception as exc:
            _log.warning("consolidation | auto-run failed: %s", exc)

    t = threading.Thread(target=_run, daemon=True, name="sofi-consolidation")
    t.start()


def _auto_register_tools(registry) -> None:
    """
    Auto-discover and register all tool modules in BRAIN/tools/.

    Any Python file in that directory that exposes a top-level function named
    'register' (callable, takes one argument: the registry) is imported and
    called automatically. Adding a new tool file requires zero changes to
    brain.py — just drop the file and implement register(registry).

    Discovery order: alphabetical by filename (consistent across runs).
    Files starting with '_' are skipped (private/test modules).
    """
    import importlib
    from pathlib import Path as _Path

    tools_dir = _Path(__file__).parent / "tools"
    registered, skipped = [], []

    for path in sorted(tools_dir.glob("*.py")):
        if path.stem.startswith("_"):
            continue  # skip __init__, _test*, etc.

        module_name = f"BRAIN.tools.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            _log.warning("_auto_register_tools | import failed | module=%s exc=%s", module_name, exc)
            skipped.append(path.stem)
            continue

        # Look for a register(registry) function
        register_fn = getattr(mod, "register", None)
        if register_fn is None or not callable(register_fn):
            continue  # no register() in this file — skip silently

        try:
            register_fn(registry)
            registered.append(path.stem)
            _log.debug("_auto_register_tools | registered | module=%s", path.stem)
        except Exception as exc:
            _log.warning(
                "_auto_register_tools | register() failed | module=%s exc=%s",
                path.stem, exc,
            )
            skipped.append(path.stem)

    _log.info(
        "_auto_register_tools | done | registered=%s skipped=%s total_tools=%d",
        registered, skipped, registry.tool_count,
    )


# Tools that take noticeable time (>~500ms). Only these trigger an ack.
# Fast tools (list_directory, read_file, get_current_time) run silently — no ack.
_SLOW_TOOLS: frozenset = frozenset({
    "web_search", "web_fetch", "get_weather",
    "run_command", "run_python",
    "write_file", "patch_file",
    "simulate_slow_search",
    "spawn_agent",
})


def _ack_for_tools(tool_calls) -> str:
    """
    Return a brief natural-language acknowledgement if this tool batch warrants one,
    or empty string if the tools are fast enough that no ack is needed.

    Rules:
    - 0 slow tools AND only 1 tool → no ack (fast path, just show tool label)
    - Any slow tool OR 2+ inline tools → ack
    Ack text is picked by dominant tool category, in SOFi's voice.
    """
    names = [tc.name for tc in tool_calls]

    slow_count = sum(1 for n in names if n in _SLOW_TOOLS)
    total_count = len(names)

    if slow_count == 0 and total_count < 2:
        return ""  # fast single tool — no ack needed

    # Pick the right ack by what's happening
    has_web    = any(n in ("web_search", "web_fetch", "get_weather") for n in names)
    has_write  = any(n in ("write_file", "patch_file") for n in names)
    has_exec   = any(n in ("run_command", "run_python") for n in names)

    if total_count >= 3:
        return "Working on it — this'll take a moment."
    if total_count >= 2:
        return "Give me a moment."
    if has_web:
        return "Checking that now."
    if has_write:
        return "On it."
    if has_exec:
        return "Running that."
    return "On it."


def _tool_display_name(tool_name: str, is_background: bool = False) -> str:
    """Human-readable tool label for inline display."""
    labels = {
        # Communication stubs
        "get_current_time":     "checking time...",
        "check_emails":         "reading emails...",
        "send_email":           "sending email...",
        "check_whatsapp":       "checking WhatsApp...",
        "send_whatsapp":        "sending WhatsApp message...",
        "check_calendar":       "checking calendar...",
        # Real web tools
        "web_search":           "searching the web...",
        "web_fetch":            "fetching page...",
        "get_weather":          "checking weather...",
        # Real filesystem tools
        "read_file":            "reading file...",
        "list_directory":       "browsing directory...",
        "search_files":         "searching files...",
        "write_file":           "writing file...",
        "patch_file":           "editing file...",
        # Real execution tools
        "run_command":          "running command...",
        "run_python":           "running Python...",
        # Sub-agents
        "spawn_agent":          "delegating to sub-agent...",
        # Skills
        "skills_list":          "checking skills...",
        "skills_load":          "loading skill...",
        # Background tools
        "write_report":         "writing report... ⟳",
        "save_note":            "saving note... ⟳",
        "log_decision":         "logging decision... ⟳",
        "create_task_item":     "adding task... ⟳",
        "simulate_slow_search": "researching in background... ⟳",
    }
    label = labels.get(tool_name, f"using {tool_name}{'... ⟳' if is_background else '...'}")
    return label


def _args_preview(args: Dict[str, Any], max_len: int = 60) -> str:
    """Compact args representation for logging."""
    if not args:
        return "()"
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > max_len:
            s = s[:max_len - 3] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
