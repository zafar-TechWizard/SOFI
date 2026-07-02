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
import threading
import time
from datetime import datetime
from pathlib import Path as _Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from memory.memory_manager import MemoryManager
from memory.working_memory.working_context import (
    NotifyPriority,
    WorkspaceItem,
    WorkspaceItemStatus,
    WorkspaceItemType,
)

import os

from BRAIN.llm import GeminiClient, GroqClient, LLMProvider
from BRAIN.context_compression import compress_loop_messages
from BRAIN.llm.error_classifier import classify_error, ErrorReason
from BRAIN.llm.retry_utils import jittered_backoff
from BRAIN.llm.sanitizer import extract_retry_after, sanitize_messages, validate_response
from BRAIN.observability.metrics import get_metrics
from BRAIN.mode import Mode, ModeController
from BRAIN.persona.persona import (
    DEFAULT_MODE,
    get_personality_dict,
    set_self_model,
    warm_cache,
)
from BRAIN.prompt import build_messages, build_prompt
from BRAIN.prompt.token_estimator import check_budget
from BRAIN.state import SelfModel, UserStateInferencer, UserStateUpdate
from BRAIN.tools.registry import ToolCall, ToolRegistry, ToolResult


_log = logging.getLogger("sofi.brain")

ProgressFn = Callable[[str], None]
ToolEventFn = Optional[Callable[[str, Dict[str, Any]], None]]
# Confirmation callback: (tool_name, question) → bool (True = approved)
ConfirmFn = Optional[Callable[[str, str], Awaitable[bool]]]


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
        self._llm: Optional[LLMProvider] = None
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
        self._confirmation_fn: ConfirmFn = None

        # Background task tracking — keeps asyncio tasks alive until done
        self._background_tasks: Set[asyncio.Task] = set()

        # Sub-agent infrastructure
        from BRAIN.agents.registry import ActiveRegistry
        from BRAIN.agents.heartbeat import HeartbeatMonitor
        self._active_registry = ActiveRegistry()
        self._heartbeat = HeartbeatMonitor(self._active_registry)

        # Proactive notification queue — populated by WorkspaceWatcher daemon thread
        # when a background agent completes with INFORM: true
        self._proactive_lock = threading.Lock()
        self._proactive_items: List[Any] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def setup(self, on_progress: Optional[ProgressFn] = None) -> None:
        """Full cold-start setup. ~30 min first run (Docker + GLiNER + Neo4j)."""
        if self._is_ready:
            return
        await self._setup_core(on_progress)
        await self._setup_memory(on_progress)
        self._preflight_check(on_progress)

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

        # LLM backend: SOFI_LLM_PROVIDERS=gemini,groq for multi-provider failover.
        # Falls back to SOFI_LLM_BACKEND=groq (legacy single-provider env var).
        # Single provider = ProviderPool with one slot; zero overhead vs. before.
        from BRAIN.llm.provider_pool import ProviderPool
        _providers_env = os.environ.get("SOFI_LLM_PROVIDERS", "").strip()
        _backend_env   = os.environ.get("SOFI_LLM_BACKEND", "").strip()
        _provider_label = _providers_env or _backend_env or "gemini"
        _tick(f"connecting LLM ({_provider_label})")
        self._llm = ProviderPool.from_env(model_override=self._model_override)

        _tick("registering tools")
        self._tool_registry = ToolRegistry()
        _auto_register_tools(self._tool_registry)
        from BRAIN.tools._agent_tools import register_agent_tools
        register_agent_tools(
            self._tool_registry,
            self._llm,
            get_workspace=self._get_workspace,
            register_bg_task=lambda t: (
                self._background_tasks.add(t),
                t.add_done_callback(self._background_tasks.discard),
            ),
            active_registry=self._active_registry,
        )
        self._heartbeat.start()
        self._tool_registry.sync_with_self_model(self._self_model)
        self._tool_registry.set_self_model(self._self_model)
        set_self_model(self._self_model)
        warm_cache()
        self._start_capability_monitor()

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

        # Wire brain's proactive callback into the memory system's WorkspaceWatcher.
        # Must happen after setup() because the watcher is created inside setup().
        # The lambda avoids a direct reference cycle: brain → memory → brain.
        _brain_ref = self
        self._memory._on_proactive_notification = lambda item: _brain_ref._on_proactive_notification(item)

        _tick("ready")
        self._is_ready = True

    def _preflight_check(self, on_progress: Optional[ProgressFn] = None) -> None:
        """
        Validate all subsystems are operational after setup.
        Logs warnings for degraded subsystems but doesn't block startup.
        """
        issues = []

        if self._llm is None:
            issues.append("LLM client not initialized")
        if self._memory is None:
            issues.append("Memory system not initialized")
        if self._self_model is None:
            issues.append("SelfModel not initialized")
        if self._tool_registry.tool_count == 0:
            issues.append("No tools registered")

        available_tools = len(self._tool_registry.get_available_tools())
        if available_tools == 0 and self._tool_registry.tool_count > 0:
            issues.append(f"All {self._tool_registry.tool_count} tools unavailable")

        if issues:
            for issue in issues:
                _log.warning("preflight | %s", issue)
            if on_progress:
                try:
                    on_progress(f"warnings: {', '.join(issues)}")
                except Exception:
                    pass
        else:
            _log.info(
                "preflight | all systems nominal | llm=%s tools=%d/%d memory=%s",
                type(self._llm).__name__,
                available_tools,
                self._tool_registry.tool_count,
                "ok" if self._memory else "none",
            )

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
            "BRAIN.llm.circuit_breaker",
            "BRAIN.llm.provider_pool",
            "BRAIN.llm",
            "BRAIN.tools.registry",
            "BRAIN.tools",
            "BRAIN.agents.budget",
            "BRAIN.agents.result",
            "BRAIN.agents.safety",
            "BRAIN.agents.registry",
            "BRAIN.agents.heartbeat",
            "BRAIN.agents.runner",
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
        new_brain._prev_user_state = None
        new_brain._last_mode_decision = None
        new_brain._prev_mode = self._prev_mode
        new_brain._prev_was_override = self._prev_was_override
        new_brain._forced_mode = self._forced_mode

        # Carry over proactive notification queue (lost on reload otherwise).
        with self._proactive_lock:
            new_brain._proactive_items = list(self._proactive_items)

        # Carry over sub-agent infrastructure — live agents must survive reload.
        new_brain._active_registry = self._active_registry
        new_brain._heartbeat = self._heartbeat
        new_brain._background_tasks = {t for t in self._background_tasks if not t.done()}

        # Carry over CLI-wired callbacks — these are closures on the terminal
        # session and console, which survive across reloads unchanged.
        new_brain._confirmation_fn = self._confirmation_fn
        new_brain._on_tool_event = self._on_tool_event

        # Run the fast path only — persona + LLM + tools (~2-3s, no Docker).
        await new_brain._setup_core()
        new_brain._is_ready = True

        _log.info(
            "hot_reload | complete | tools=%d",
            new_brain._tool_registry.tool_count,
        )
        return new_brain

    async def shutdown(self) -> None:
        # Stop heartbeat monitor and interrupt all active sub-agents
        self._heartbeat.stop()
        self._active_registry.interrupt_all()

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

        # Clean up old .temp/ files from sub-agent outputs.
        _cleanup_temp_files()

        # Clean up old task files on disk.
        tm = self._get_task_manager()
        if tm:
            try:
                tm.cleanup()
            except Exception:
                pass

        # Auto-run consolidation in background on shutdown so memories accumulate
        # without Zafar having to remember to run it manually.
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
        _turn_start = time.perf_counter()

        self._local_history.append({"role": "user", "content": message})
        self._local_history = self._local_history[-self.MAX_LOCAL_TURNS:]

        # ─── PHASE A: Context gathering ───
        # Wrapped in try/except so SOFi degrades to conversation-only if
        # memory is unavailable (Neo4j down, GLiNER crash, Docker stopped).
        _memory_ok = True
        try:
            await self._memory.observe("user", message)
            await self._memory.get_context_async("user", message)
            ctx = self._memory.get_full_context()
        except Exception as mem_exc:
            _log.warning(
                "process | memory unavailable — degrading to conversation-only | %s",
                mem_exc,
            )
            _memory_ok = False
            ctx = None

        # ─── PHASE B: State + Mode inference ───
        if _memory_ok and ctx is not None:
            user_state = self._user_state_inferencer.infer(
                ctx, message, prev_state=self._prev_user_state,
            )
            try:
                self._memory.context_manager.update_user_state(**user_state.as_dict())
                ctx = self._memory.get_full_context()
            except Exception:
                pass
        else:
            user_state = self._user_state_inferencer.infer(
                None, message, prev_state=self._prev_user_state,
            )
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

        # Lazy-import skills registry — avoids circular import and only loads
        # if the module is available.
        _skills_reg = None
        try:
            from BRAIN.skills._registry import get_registry
            _skills_reg = get_registry()
        except Exception:
            pass

        system_prompt = build_prompt(
            ctx,
            mode=decision.mode.value,
            allow_dropped_formality=decision.allow_dropped_formality,
            action_state=action_state,
            self_model=self._self_model,
            skills_registry=_skills_reg,
        )

        if _memory_ok and ctx is not None:
            messages = build_messages(ctx, message)
            memory_recent = getattr(getattr(ctx, "memory", None), "recent_turns", None)
            if not memory_recent:
                messages = list(self._local_history)
        else:
            messages = list(self._local_history)

        tool_defs = self._tool_registry.get_definitions()

        # ─── Token budget check ───
        # Trim oldest messages (keeping the current user message) if the
        # prompt + messages exceed the context window.
        fits, est_tokens, _ = check_budget(system_prompt, messages)
        if not fits and len(messages) > 2:
            _log.warning(
                "process | token budget exceeded (%d est.) — trimming messages",
                est_tokens,
            )
            while not fits and len(messages) > 2:
                messages.pop(0)
                fits, est_tokens, _ = check_budget(system_prompt, messages)

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
        _continuation_count = 0
        _exit_reason = "normal"
        _tools_executed = 0
        _loop_start = time.perf_counter()
        _spill_dir = _Path(__file__).parent.parent / ".temp"

        MAX_CONTINUATIONS = 3
        MAX_EMPTY_RETRIES = 2
        PROGRESSIVE_MAX_TOKENS = [8192, 12288, 16384]
        _empty_retry_count = 0

        if not tool_defs:
            _log.debug("process | no tools registered — pure conversation stream")
            async for token in self._llm.stream(system_prompt, messages):
                response_text += token
                yield token
            if not response_text:
                _log.warning("process | stream returned no tokens")
                fallback = "I didn't get a response from the model. Try again, sir."
                response_text = fallback
                yield fallback
        else:
            _log.debug("process | agentic loop start | tools=%d", len(tool_defs))

            while iteration < self.MAX_TOOL_ITERATIONS:
                iteration += 1
                _log.debug("process | agentic loop iter=%d", iteration)

                # ── Per-iteration budget check ──
                messages = compress_loop_messages(
                    messages,
                    system_prompt_chars=len(system_prompt),
                )

                # ── Grace call on last iteration ──
                current_tool_defs = tool_defs
                if iteration >= self.MAX_TOOL_ITERATIONS:
                    _log.info("process | grace call — last iteration, removing tools")
                    messages.append({
                        "role": "system",
                        "content": (
                            "This is your last iteration. Give a complete response "
                            "now. Do not call more tools."
                        ),
                    })
                    current_tool_defs = []

                # ── Sanitize messages before LLM call ──
                messages = sanitize_messages(messages)

                # ── LLM call with retry ──
                max_tok = PROGRESSIVE_MAX_TOKENS[
                    min(_continuation_count, len(PROGRESSIVE_MAX_TOKENS) - 1)
                ] if _continuation_count > 0 else None

                response, error_msg = await self._call_llm_with_retry(
                    system_prompt, messages, current_tool_defs,
                    max_tokens_override=max_tok,
                )

                if error_msg is not None:
                    _exit_reason = "llm_error"
                    error_text = f"\n\n{error_msg}"
                    response_text += error_text
                    yield error_text
                    break

                # ── Post-parse validation (drops invalid tool calls, fixes types) ──
                validate_response(response)

                # ── CASE: finish_reason == "error" (empty response from provider) ──
                if response.finish_reason == "error":
                    _empty_retry_count += 1
                    if _empty_retry_count <= MAX_EMPTY_RETRIES and iteration < self.MAX_TOOL_ITERATIONS:
                        _log.warning(
                            "process | empty response from provider — retrying (%d/%d)",
                            _empty_retry_count, MAX_EMPTY_RETRIES,
                        )
                        await asyncio.sleep(1.0)
                        continue
                    _exit_reason = "empty_provider_response"
                    _log.warning(
                        "process | empty response — giving up after %d retries",
                        _empty_retry_count,
                    )
                    break

                # ── CASE: finish_reason == "length" ──
                if response.finish_reason == "length":
                    if response.text:
                        response_text += response.text
                        yield response.text

                    _continuation_count += 1
                    if _continuation_count >= MAX_CONTINUATIONS:
                        _log.info(
                            "process | max continuations reached (%d) — stopping",
                            MAX_CONTINUATIONS,
                        )
                        _exit_reason = "max_continuations"
                        break

                    _log.debug(
                        "process | finish_reason=length — continuing (%d/%d) max_tokens=%s",
                        _continuation_count, MAX_CONTINUATIONS,
                        PROGRESSIVE_MAX_TOKENS[min(_continuation_count, len(PROGRESSIVE_MAX_TOKENS) - 1)],
                    )
                    messages.append({"role": "assistant", "content": response.text or ""})
                    messages.append({"role": "user", "content": "Please continue your response."})
                    continue

                # ── CASE: content filter ──
                if response.finish_reason == "content_filter":
                    _exit_reason = "content_filter"
                    _log.warning("process | content filter stop")
                    break

                # ── CASE: tool_calls present ──
                if response.tool_calls:
                    if response.text:
                        _log.debug(
                            "process | suppressing LLM reasoning text alongside tools | "
                            "len=%d preview=%.80s", len(response.text), response.text,
                        )

                    ack_text = _ack_for_tools(response.tool_calls)
                    if ack_text and iteration == 1:
                        display_text += ack_text
                        response_text += ack_text
                        yield ack_text

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

                    # ── Phase 1: Show labels + dispatch background tools ──
                    had_inline = False
                    inline_tcs = []
                    bg_dispatched: Dict[str, Any] = {}

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

                    # ── Pre-flight safety check ──
                    pre_blocked: Dict[str, ToolResult] = {}
                    if inline_tcs:
                        inline_tcs, pre_blocked = await self._pre_flight_check(inline_tcs)

                    # ── Phase 2: Execute all inline tools in parallel ──
                    inline_results: Dict[str, Any] = {}
                    inline_results.update(pre_blocked)

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
                            _tools_executed += 1
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

                    # ── Phase 3: Append tool messages in order ──
                    tool_msg_start_idx = len(messages)
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

                    # ── Turn budget enforcement (spill oversized results to disk) ──
                    from BRAIN.tools.turn_budget import enforce_turn_budget
                    messages = enforce_turn_budget(
                        messages, tool_msg_start_idx, _spill_dir,
                    )

                    if had_inline:
                        _log.debug("process | had inline tools — continuing loop")
                        continue
                    else:
                        ack = "On it."
                        response_text += ack
                        yield ack
                        _exit_reason = "all_background"
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
                _exit_reason = "text_response"
                break

            # Safety: loop exited with no text response
            if not response_text:
                if iteration >= self.MAX_TOOL_ITERATIONS:
                    _exit_reason = "max_iterations"
                    _log.warning("process | max_iterations=%d hit without text response", self.MAX_TOOL_ITERATIONS)
                    fallback = "I seem to have gotten caught in a loop, sir. Let me answer directly."
                else:
                    _exit_reason = "empty_response"
                    _log.warning("process | LLM returned empty response on iteration %d", iteration)
                    fallback = "I didn't get a response from the model. Try again, sir."
                response_text = fallback
                yield fallback

            # ── Loop summary log ──
            _loop_duration = (time.perf_counter() - _loop_start) * 1000
            _log.info(
                "process | loop done | exit=%s iterations=%d tools=%d "
                "continuations=%d duration_ms=%.0f",
                _exit_reason, iteration, _tools_executed,
                _continuation_count, _loop_duration,
            )

        # ─── PHASE E: Post-response ───
        # Save only the LLM-generated text to memory (not tool status markers)
        if response_text:
            if _memory_ok:
                try:
                    await self._memory.observe("assistant", response_text)
                except Exception as mem_exc:
                    _log.warning("process | failed to save response to memory | %s", mem_exc)
            self._local_history.append({"role": "assistant", "content": response_text})
            self._local_history = self._local_history[-self.MAX_LOCAL_TURNS:]
            _log.debug("process | response saved to memory | len=%d", len(response_text))

        # ── SofiState update (post-response, rule-based, ~0.1ms) ──
        if _memory_ok and response_text:
            self._update_sofi_state(decision, response_text, user_state)

        # ── Response analysis (fire-and-forget, ~1ms in background) ──
        if _memory_ok and response_text:
            _t = asyncio.create_task(
                self._analyze_response(response_text),
                name="sofi-response-analyzer",
            )
            self._background_tasks.add(_t)
            _t.add_done_callback(self._background_tasks.discard)

        # ── Metrics (fire-and-forget, sub-µs) ──
        _m = get_metrics()
        _m.inc("turns")
        _m.observe("turn_latency_ms", (time.perf_counter() - _turn_start) * 1000)
        if response_text:
            _m.observe("response_chars", len(response_text))
        _m.flush()

        # ─── PHASE F: Mark delivered tasks ───
        # If there were completed deliveries in the action_state and SOFi
        # generated a response (meaning she delivered them), mark as delivered.
        if response_text and action_state:
            deliveries = action_state.get("deliveries") or []
            tm = self._get_task_manager()
            if tm and deliveries:
                for d in deliveries:
                    task_id = d.get("task_id")
                    if task_id:
                        tm.mark_delivered(task_id)
                        _log.info("process | task marked delivered | id=%s", task_id)

    async def process_proactive(self, item) -> AsyncIterator[str]:
        """
        SOFi speaks unprompted to announce a completed background task.

        Called by the CLI when an URGENT WorkspaceItem arrives during an idle prompt.
        Skips the memory observe for the synthetic trigger (no meaningful user message),
        but saves SOFi's response as a normal assistant turn.

        Yields the same token stream as process() — CLI renders it identically.
        """
        self._require_ready()
        _turn_start = time.perf_counter()

        item_title = getattr(item, "title", "") or ""
        item_desc  = getattr(item, "description", "") or ""
        meta       = getattr(item, "metadata", {}) or {}

        # Build a synthetic message so memory context + recent turns are available,
        # but we don't add it as a real user turn.
        synthetic = f"[PROACTIVE] {item_title}: {item_desc[:200]}"

        _memory_ok = True
        try:
            await self._memory.get_context_async("assistant", synthetic)
            ctx = self._memory.get_full_context()
        except Exception as mem_exc:
            _log.warning("process_proactive | memory unavailable | %s", mem_exc)
            _memory_ok = False
            ctx = None

        from BRAIN.mode.controller import ModeDecision
        decision = ModeDecision(
            mode=Mode.CONVERSATIONAL,
            allow_dropped_formality=False,
            scores={},
            triggered_overrides=["proactive"],
            held_prev=False,
        )

        system_prompt = build_prompt(
            ctx,
            mode=decision.mode.value,
            allow_dropped_formality=False,
            is_proactive=True,
            proactive_title=item_title,
        )

        messages = []
        if _memory_ok and ctx is not None:
            messages = build_messages(ctx, synthetic)
        else:
            messages = list(self._local_history)
        # Replace the synthetic message with a clean delivery-context message
        # so the LLM knows WHAT to announce without polluting history.
        if messages and messages[-1].get("content", "").startswith("[PROACTIVE]"):
            messages[-1]["content"] = (
                f"Background task completed — announce it briefly:\n"
                f"Task: {item_title}\n"
                f"Result: {item_desc[:300]}"
            )

        response_text = ""
        try:
            async for token in self._llm.stream(system_prompt, messages):
                response_text += token
                yield token
        except Exception as exc:
            _log.warning("process_proactive | stream error | %s", exc)
            fallback = f"That background task finished, sir. {item_title}."
            response_text = fallback
            yield fallback

        # Save proactive response to memory as assistant turn
        if response_text and _memory_ok:
            try:
                await self._memory.observe("assistant", response_text)
            except Exception:
                pass

        self._local_history.append({"role": "assistant", "content": response_text})
        self._local_history = self._local_history[-self.MAX_LOCAL_TURNS:]

        _m = get_metrics()
        _m.inc("proactive_turns")
        _m.observe("turn_latency_ms", (time.perf_counter() - _turn_start) * 1000)
        _m.flush()

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

    def _on_proactive_notification(self, item) -> None:
        """
        Called from WorkspaceWatcher daemon thread when a background agent
        completes with INFORM: true. Thread-safe — stores item for CLI pickup.
        """
        _log.info(
            "proactive | queued | id=%s title=%s",
            item.id[:8], item.title,
        )
        with self._proactive_lock:
            self._proactive_items.append(item)

    def get_pending_proactive(self) -> list:
        """
        Non-blocking, thread-safe drain of pending proactive notifications.
        Called by the CLI at the start of each main loop iteration.
        """
        with self._proactive_lock:
            items = list(self._proactive_items)
            self._proactive_items.clear()
            return items

    # =========================================================================
    # Tool event callback — wired by CLI for live display
    # =========================================================================

    def set_tool_event_handler(self, handler: ToolEventFn) -> None:
        self._on_tool_event = handler

    def set_confirmation_handler(self, handler: ConfirmFn) -> None:
        """Wire the CLI's confirmation prompt into the agentic loop."""
        self._confirmation_fn = handler

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
            "metrics": get_metrics().snapshot(),
        }
        # Provider pool health (circuit breaker states per provider)
        from BRAIN.llm.provider_pool import ProviderPool
        if isinstance(self._llm, ProviderPool):
            snap["providers"] = self._llm.health_report()

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

    def _get_task_manager(self):
        """Get the disk-backed TaskManager from the tool registry."""
        return getattr(self._tool_registry, "_task_manager", None)

    def _get_action_state(self) -> Optional[Dict[str, Any]]:
        """
        Assemble the action state dict for prompt injection.

        Reads from THREE sources:
          1. self._last_tool_calls — inline tools that ran last turn
          2. AgenticWorkspace     — in-memory background task state
          3. TaskManager (disk)   — task files with step progress + deliveries

        Task deliveries from disk are the primary source for completed work.
        They include the full content that SOFi needs to deliver to Zafar.
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

        # ── Disk-backed tasks (primary source for delegated work) ──
        tm = self._get_task_manager()
        if tm:
            # Active tasks with step-level progress
            active_tasks = tm.get_active_tasks()
            if active_tasks:
                state["active_tasks"] = []
                for t in active_tasks[:5]:
                    entry: Dict[str, Any] = {
                        "task_id": t.task_id,
                        "agent": t.agent_type,
                        "query": t.original_query[:120],
                        "status": t.status,
                    }
                    if t.total_steps > 0:
                        entry["progress"] = f"step {t.current_step + 1}/{t.total_steps}"
                        current = t.steps[t.current_step] if t.current_step < len(t.steps) else None
                        if current:
                            entry["current_action"] = current.get("action", "")
                            entry["detail"] = current.get("detail", "")[:100]
                    state["active_tasks"].append(entry)

            # Completed deliveries — SOFi must deliver these to Zafar
            ready = tm.get_completed_undelivered()
            if ready:
                state["deliveries"] = []
                for t in ready[:3]:
                    delivery = t.delivery or {}
                    state["deliveries"].append({
                        "task_id": t.task_id,
                        "agent": t.agent_type,
                        "original_query": t.original_query,
                        "delivery_status": delivery.get("status", "unknown"),
                        "summary": delivery.get("summary", ""),
                        "content": delivery.get("content", ""),
                        "gaps": delivery.get("gaps"),
                    })
                _log.info(
                    "_get_action_state | %d deliveries ready: %s",
                    len(ready),
                    [t.task_id for t in ready],
                )

            # Recently delivered — content SOFi already presented but may
            # need to reference again (e.g. "write that to a file").
            # Shows summary + content so SOFi can act on follow-up requests.
            recent = tm.get_recently_delivered(max_age_minutes=30)
            if recent:
                state["recent_deliveries"] = []
                for t in recent[:3]:
                    delivery = t.delivery or {}
                    state["recent_deliveries"].append({
                        "task_id": t.task_id,
                        "agent": t.agent_type,
                        "original_query": t.original_query,
                        "summary": delivery.get("summary", ""),
                        "content": delivery.get("content", ""),
                    })

        # ── Active sub-agents (real-time from registry) ──
        if self._active_registry.active_count() > 0:
            snap = self._active_registry.snapshot()
            state["live_agents"] = snap["active"]
            state["agent_slots"] = snap["slots"]

        # ── AgenticWorkspace: legacy background tasks ──
        ws = self._get_workspace()
        if ws:
            notifications = ws.get_pending_notifications()
            if notifications:
                state["notifications"] = [
                    {
                        "summary": (
                            f"{item.title} — "
                            f"{'done' if item.status == WorkspaceItemStatus.COMPLETED else 'failed'}: "
                            f"{item.description[:150]}"
                        ),
                        "agent_type": (item.metadata or {}).get("agent_type", ""),
                    }
                    for item in notifications[:3]
                ]
                for item in notifications:
                    ws.update_item(item.id, notify=False)

        if not state:
            return None

        _log.debug("_get_action_state | state_keys=%s", list(state.keys()))
        return state

    async def _call_llm_with_retry(
        self,
        system_prompt: str,
        messages: List[Dict],
        tool_defs: List[Dict],
        max_tokens_override: Optional[int] = None,
    ) -> Tuple[Any, Optional[str]]:
        """
        Call LLM with structured retry and context-overflow recovery.

        Returns (LLMResponse, error_msg). error_msg is None on success.
        On non-recoverable errors, LLMResponse is None and error_msg is set.
        """
        MAX_ATTEMPTS = 3

        for attempt in range(MAX_ATTEMPTS):
            try:
                get_metrics().inc("llm_calls")
                response = await self._llm.call_with_tools(
                    system_prompt, messages, tool_defs,
                    max_tokens_override=max_tokens_override,
                )
                return response, None

            except Exception as exc:
                classified = classify_error(exc)
                get_metrics().inc("llm_errors")
                get_metrics().inc(f"llm_error_{classified.reason.value}")
                _log.warning(
                    "_call_llm_with_retry | attempt=%d reason=%s retryable=%s | %s",
                    attempt + 1, classified.reason.value, classified.retryable, exc,
                )

                if not classified.retryable:
                    return None, classified.user_message

                if classified.should_trim_context:
                    messages = compress_loop_messages(
                        messages,
                        system_prompt_chars=len(system_prompt),
                    )
                    continue

                if classified.reason in (
                    ErrorReason.RATE_LIMIT, ErrorReason.OVERLOADED,
                ):
                    retry_after = extract_retry_after(exc)
                    delay = retry_after if retry_after else jittered_backoff(attempt, base_delay=3.0)
                elif classified.reason in (
                    ErrorReason.SERVER_ERROR, ErrorReason.TIMEOUT,
                ):
                    delay = jittered_backoff(attempt, base_delay=1.0)
                else:
                    delay = jittered_backoff(attempt, base_delay=2.0)

                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(delay)
                else:
                    return None, classified.user_message

        return None, "All retry attempts exhausted."

    async def _pre_flight_check(
        self,
        tcs: List[ToolCall],
    ) -> Tuple[List[ToolCall], Dict[str, ToolResult]]:
        """
        Safety gate run BEFORE inline tool execution.

        For each tool call:
          - Hardline patterns → blocked unconditionally (no confirmation offered)
          - Dangerous patterns → blocked if confirmation_fn is set AND user says no
          - needs_confirmation flag → confirmed if confirmation_fn is set
          - Everything else → approved

        Returns (approved_tcs, blocked_results) where blocked_results maps
        tc.id → ToolResult for tools that were blocked or declined.
        """
        from BRAIN.tools.exec_tools import check_command_safety, check_python_safety

        approved: List[ToolCall] = []
        blocked: Dict[str, ToolResult] = {}

        for tc in tcs:
            tier, reason = "safe", ""

            if tc.name == "run_command":
                tier, reason = check_command_safety(tc.arguments.get("command", ""))
            elif tc.name == "run_python":
                tier, reason = check_python_safety(tc.arguments.get("code", ""))
            elif self._tool_registry.needs_confirmation(tc.name):
                tier, reason = "dangerous", f"{tc.name} requires confirmation"

            if tier == "blocked":
                _log.warning(
                    "pre_flight | hardline block | tool=%s reason=%s", tc.name, reason
                )
                blocked[tc.id] = ToolResult(
                    success=False,
                    output="",
                    error=f"Blocked — this operation is not allowed: {reason}.",
                )
                continue

            if tier == "dangerous":
                if self._confirmation_fn is not None:
                    question = self._build_confirm_question(tc, reason)
                    try:
                        confirmed = await self._confirmation_fn(tc.name, question)
                    except Exception as exc:
                        _log.warning("pre_flight | confirmation callback error: %s", exc)
                        confirmed = False

                    if not confirmed:
                        _log.info(
                            "pre_flight | declined by user | tool=%s reason=%s",
                            tc.name, reason,
                        )
                        blocked[tc.id] = ToolResult(
                            success=False,
                            output="",
                            error="Declined by Zafar — not executed.",
                        )
                        continue
                    _log.info(
                        "pre_flight | confirmed by user | tool=%s", tc.name
                    )
                else:
                    _log.warning(
                        "pre_flight | dangerous tool with no confirmation_fn — "
                        "running anyway | tool=%s reason=%s", tc.name, reason,
                    )

            approved.append(tc)

        return approved, blocked

    def _build_confirm_question(self, tc: ToolCall, reason: str) -> str:
        """Generate a short, specific confirmation question in SOFi's voice."""
        if tc.name == "run_command":
            cmd = tc.arguments.get("command", "")
            short = cmd[:70] + ("…" if len(cmd) > 70 else "")
            return f"`{short}` — {reason}. Confirm?"
        if tc.name == "run_python":
            return f"Python code contains {reason}. Confirm?"
        return f"{tc.name} — {reason}. Confirm?"

    def _update_sofi_state(self, decision, response_text: str, user_state) -> None:
        """
        Post-response SofiState enrichment. Rule-based, ~0.1ms.
        Wrapped in try/except — enrichment, not correctness.
        """
        try:
            _TONE_MAP = {
                Mode.CONVERSATIONAL: "calm",
                Mode.FOCUSED: "focused",
                Mode.CREATIVE: "playful",
            }
            if decision.mode == Mode.EMPATHETIC:
                tone = "warm" if (user_state.emotional_intensity or 0) > 0.5 else "concerned"
            else:
                tone = _TONE_MAP.get(decision.mode, "neutral")

            resp_len = len(response_text)
            energy = "high" if resp_len > 500 else ("low" if resp_len < 50 else "normal")

            current_focus = ""
            try:
                ctx = self._memory.get_full_context()
                if ctx and hasattr(ctx, "user"):
                    entities = getattr(ctx.user, "mentioned_entities", None)
                    if entities:
                        current_focus = entities[0] if isinstance(entities, list) else str(entities)
            except Exception:
                pass

            self._memory.context_manager.update_sofi_state(
                emotional_tone=tone,
                energy_level=energy,
                current_mode=decision.mode.value,
                current_focus=current_focus,
            )
        except Exception as exc:
            _log.debug("_update_sofi_state | failed (non-critical): %s", exc)

    async def _analyze_response(self, response_text: str) -> None:
        """
        Post-response analysis — extracts topics, commitments, questions from
        SOFi's own output and writes them to SofiState for next-turn awareness.
        Fire-and-forget: runs as asyncio.Task, never delays streaming.
        """
        try:
            from BRAIN.state.response_analyzer import ResponseAnalyzer
            analysis = ResponseAnalyzer().analyze(response_text)
            self._memory.context_manager.update_sofi_response_state(
                last_topics_discussed=analysis.topics,
                last_commitments=analysis.commitments,
                last_questions_asked=analysis.questions,
            )
            _log.debug(
                "_analyze_response | topics=%d commitments=%d questions=%d",
                len(analysis.topics), len(analysis.commitments), len(analysis.questions),
            )
        except Exception as exc:
            _log.debug("_analyze_response | failed (non-critical): %s", exc)

    def _start_capability_monitor(self) -> None:
        """
        Daemon thread that re-checks each tool's check_fn every 60s and
        updates SelfModel availability state on change.

        This ensures SOFi's self-awareness stays current without any latency
        impact on the response path.
        """
        import time as _time

        def _monitor_loop() -> None:
            while True:
                _time.sleep(60)
                try:
                    self._check_capabilities_once()
                except Exception as exc:
                    _log.debug("capability_monitor | error: %s", exc)

        t = threading.Thread(
            target=_monitor_loop,
            name="sofi-capability-monitor",
            daemon=True,
        )
        t.start()
        _log.debug("capability_monitor | started (60s interval)")

    def _check_capabilities_once(self) -> None:
        """Check all tools with check_fn and update SelfModel on state change."""
        if self._self_model is None:
            return
        changed = 0
        for entry in self._tool_registry._tools.values():
            if entry.check_fn is None:
                continue
            cap_name = entry.capability_name or entry.name
            cap = self._self_model.get(cap_name)
            if cap is None:
                continue
            try:
                now_available = bool(entry.check_fn())
            except Exception:
                now_available = False
            if cap.available != now_available:
                cap.available = now_available
                changed += 1
                _log.info(
                    "capability_monitor | state change | cap=%s available=%s",
                    cap_name, now_available,
                )
        if changed:
            set_self_model(self._self_model)
            warm_cache()
            _log.debug("capability_monitor | %d capability change(s) — cache refreshed", changed)

    def _require_ready(self) -> None:
        if not self._is_ready or self._llm is None or self._memory is None:
            raise RuntimeError(
                "Brain is not initialized. Call `await brain.setup()` first."
            )


# =============================================================================
# Helpers
# =============================================================================

def _cleanup_temp_files(max_age_hours: int = 24, max_files: int = 50) -> None:
    """
    Remove old .temp/ files from sub-agent outputs.

    Keeps the directory from growing unbounded. Runs synchronously on
    shutdown — fast enough for dozens of files.
    """
    from pathlib import Path as _P
    import time as _time

    temp_dir = _P(__file__).parent.parent / ".temp"
    if not temp_dir.exists():
        return

    cutoff = _time.time() - (max_age_hours * 3600)
    files = sorted(temp_dir.glob("*.md"), key=lambda f: f.stat().st_mtime)

    removed = 0
    for f in files:
        try:
            if f.stat().st_mtime < cutoff or len(files) - removed > max_files:
                f.unlink()
                removed += 1
        except Exception:
            pass

    if removed:
        _log.info("temp_cleanup | removed %d old file(s) from .temp/", removed)


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

        # Pattern A: register(registry) function — for complex / closure-based tools.
        register_fn = getattr(mod, "register", None)
        if register_fn is not None and callable(register_fn):
            try:
                register_fn(registry)
                registered.append(path.stem)
                _log.debug("_auto_register_tools | register() | module=%s", path.stem)
            except Exception as exc:
                _log.warning(
                    "_auto_register_tools | register() failed | module=%s exc=%s",
                    path.stem, exc,
                )
                skipped.append(path.stem)

        # Pattern B: @tool-decorated functions — ToolEntry stored as fn._tool_entry.
        # Both patterns can coexist in the same file.
        for attr_name in dir(mod):
            try:
                obj = getattr(mod, attr_name)
            except Exception:
                continue
            entry = getattr(obj, "_tool_entry", None)
            if entry is None:
                continue
            try:
                registry.register(entry)
                if path.stem not in registered:
                    registered.append(path.stem)
                _log.debug(
                    "_auto_register_tools | @tool | module=%s name=%s",
                    path.stem, entry.name,
                )
            except Exception as exc:
                _log.warning(
                    "_auto_register_tools | @tool failed | module=%s name=%s exc=%s",
                    path.stem, attr_name, exc,
                )

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
        "spawn_agent":          "working on it...",
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
