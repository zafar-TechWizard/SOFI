"""
BRAIN/ui/cli.py — Terminal UI for SOFi

Stack: rich (output, live streaming, panels) + prompt_toolkit (input).

The CLI knows about user input, token streaming, slash commands, and
diagnostic display. It does NOT know about Groq, persona, memory, or modes.
Those live behind the `process` / `clear_history` / `inspector` callables
passed in.

Slash commands:
    /exit              quit cleanly
    /clear             clear in-session conversation (memory graph untouched)
    /mode <name>       force mode: conversational | empathetic | focused | creative
    /mode auto         return to controller-driven mode
    /status            show full current state snapshot
    /memory            show memories surfaced on the last turn
    /tools             show registered tools (inline + background)
    /workspace         show AgenticWorkspace — active and completed background tasks
    /help              show this help
"""

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ============================================================================
# Types — public callable contracts the CLI consumes
# ============================================================================

ProcessFn = Callable[[str], AsyncIterator[str]]
ClearFn = Callable[[], None]

InspectorFn    = Optional[Callable[[], Dict[str, Any]]]
SetModeFn      = Optional[Callable[[str], None]]
ToolStatusFn   = Optional[Callable[[], Dict[str, Any]]]
ToolCallsFn    = Optional[Callable[[], List[Dict[str, Any]]]]
ReloadFn       = Optional[Callable[[], Any]]   # async: () -> new Brain (or None)
# (tool_name: str, question: str) → bool; wired ONCE after PromptSession is created
SetConfirmFn   = Optional[Callable[[Callable], None]]


# ============================================================================
# Slash-command autocomplete
# ============================================================================

class SlashCommandCompleter(Completer):
    """
    Dropdown completer that activates only when input starts with '/'.
    Shows all available slash commands with descriptions.
    Navigate with ↑↓ and confirm with Tab or Enter.
    """

    _COMMANDS = [
        ("/exit",                  "quit SOFi"),
        ("/quit",                  "quit SOFi"),
        ("/q",                     "quit"),
        ("/clear",                 "clear in-session conversation"),
        ("/reload",                "hot-reload BRAIN code (~3s, memory untouched)"),
        ("/mode conversational",   "force conversational mode"),
        ("/mode empathetic",       "force empathetic mode"),
        ("/mode focused",          "force focused / task mode"),
        ("/mode creative",         "force creative mode"),
        ("/mode auto",             "return to controller-driven mode"),
        ("/status",                "show full state snapshot"),
        ("/memory",                "show memories surfaced last turn"),
        ("/tools",                 "show registered tools"),
        ("/workspace",             "show background task status"),
        ("/help",                  "show help"),
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        text_lower = text.lower()
        for cmd, meta in self._COMMANDS:
            if cmd.startswith(text_lower):
                # Yield only the remaining suffix so the typed portion stays.
                # e.g. user typed "/st" → cmd="/status" → yield "atus"
                yield Completion(
                    cmd[len(text):],
                    start_position=0,
                    display=cmd,
                    display_meta=meta,
                )


# ============================================================================
# Persona-coloured cold-start greeting
# ============================================================================

def _greeting_for_time(now_hour: Optional[int] = None) -> str:
    """Short Jarvis-coded greeting for the boot banner."""
    if now_hour is None:
        import datetime as _dt
        now_hour = _dt.datetime.now().hour
    if 5 <= now_hour < 12:
        return "Good morning, sir."
    if 12 <= now_hour < 17:
        return "Afternoon, sir."
    if 17 <= now_hour < 22:
        return "Evening, sir."
    return "Late one, sir."


# ============================================================================
# Renderer
# ============================================================================

async def _render_stream(
    console: Console,
    token_stream: AsyncIterator[str],
    inspector: InspectorFn = None,
    tool_calls_fn: ToolCallsFn = None,
) -> str:
    """
    Render the token stream live as Markdown and return the accumulated text.
    Shows tool execution info inline before the response panel.
    """
    accumulated = ""
    t0 = time.perf_counter()
    first_token_ms: float | None = None
    _tool_events: List[Dict[str, Any]] = []

    with Live(
        Panel("", title="SOFi", title_align="left", border_style="cyan"),
        console=console,
        refresh_per_second=20,
        transient=False,
    ) as live:
        async for token in token_stream:
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t0) * 1000
            accumulated += token
            live.update(
                Panel(
                    Markdown(accumulated),
                    title="SOFi",
                    title_align="left",
                    border_style="cyan",
                )
            )

    total_ms = (time.perf_counter() - t0) * 1000

    # Show tool calls that happened during this turn (Claude Code style)
    if tool_calls_fn:
        calls = tool_calls_fn()
        if calls:
            tool_parts = []
            for tc in calls:
                icon = "[green]✓[/green]" if tc.get("success") else "[red]✗[/red]"
                dur = tc.get("duration_ms", 0)
                tool_parts.append(
                    f"  {icon} [cyan]{tc['name']}[/cyan] [dim]({dur:.0f}ms)[/dim]"
                )
            console.print(
                f"[dim]── tools ({len(calls)}) ──[/dim]\n" + "\n".join(tool_parts)
            )

    status_line = _build_status_line(inspector) if inspector else None

    diag_parts = []
    if status_line:
        diag_parts.append(status_line)
    if tool_calls_fn:
        calls = tool_calls_fn()
        if calls:
            total_tool_ms = sum(tc.get("duration_ms", 0) for tc in calls)
            diag_parts.append(f"tools={len(calls)}/{total_tool_ms:.0f}ms")
    if first_token_ms is not None:
        diag_parts.append(f"first-token {first_token_ms:.0f}ms")
    diag_parts.append(f"total {total_ms:.0f}ms")
    console.print(f"[dim]{' · '.join(diag_parts)}[/dim]")
    return accumulated


# ============================================================================
# Slash command handlers
# ============================================================================

def _print_help(console: Console) -> None:
    console.print(Panel(
        "[bold]Commands[/bold]\n\n"
        "  [cyan]/exit[/cyan]              quit\n"
        "  [cyan]/clear[/cyan]             clear in-session conversation (memory graph untouched)\n"
        "  [cyan]/reload[/cyan]            hot-reload all BRAIN code (~3s, memory stays live)\n"
        "  [cyan]/mode <name>[/cyan]       force mode: conversational | empathetic | focused | creative\n"
        "  [cyan]/mode auto[/cyan]         return to controller-driven mode\n"
        "  [cyan]/status[/cyan]            show full current state snapshot\n"
        "  [cyan]/memory[/cyan]            show memories surfaced on the last turn\n"
        "  [cyan]/tools[/cyan]             show registered tools (inline + background ⟳)\n"
        "  [cyan]/workspace[/cyan]         show AgenticWorkspace — background task status\n"
        "  [cyan]/help[/cyan]              this help",
        title="Help",
        border_style="dim",
    ))


def _print_status(console: Console, inspector: InspectorFn) -> None:
    if inspector is None:
        console.print("[red]status not available — no inspector wired.[/red]")
        return
    snap = inspector() or {}

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()

    mode_info = snap.get("mode_decision") or {}
    user_state = snap.get("user_state") or {}

    t.add_row("mode",        f"[bold]{snap.get('mode', '?')}[/bold]")
    if mode_info.get("triggered_overrides"):
        t.add_row("triggers", ", ".join(mode_info["triggered_overrides"]))
    if mode_info.get("held_prev"):
        t.add_row("held_prev", "yes (margin gate)")
    if mode_info.get("scores"):
        scores = sorted(
            mode_info["scores"].items(), key=lambda kv: kv[1], reverse=True
        )
        t.add_row("scores", ", ".join(f"{k}={v}" for k, v in scores))

    t.add_row("", "")  # spacer
    t.add_row("emotion",    str(user_state.get("current_emotional_state", "—")))
    t.add_row("intensity",  str(user_state.get("emotional_intensity", "—")))
    t.add_row("need",       str(user_state.get("current_need", "—")))
    t.add_row("engagement", str(user_state.get("engagement_level", "—")))

    sofi = snap.get("sofi") or {}
    if sofi:
        t.add_row("", "")
        t.add_row("time",        str(sofi.get("current_datetime", "—")))
        t.add_row("time of day", str(sofi.get("time_of_day", "—")))

    tools = snap.get("tools") or {}
    if tools:
        n_bg = sum(1 for i in tools.values() if i.get("background"))
        t.add_row("", "")
        t.add_row(
            "tools",
            f"{tools.get('available', 0)}/{tools.get('registered', 0)} available "
            f"({n_bg} background ⟳)"
        )

    last_calls = snap.get("last_tool_calls") or []
    if last_calls:
        t.add_row("last tools", ", ".join(tc.get("name", "?") for tc in last_calls[-3:]))

    bg_active = snap.get("background_tasks_active", 0)
    if bg_active:
        t.add_row("", "")
        t.add_row("bg tasks", f"[yellow]{bg_active} running[/yellow]")

    workspace = snap.get("workspace") or []
    pending = [w for w in workspace if w.get("status") in ("pending", "in_progress")]
    completed = [w for w in workspace if w.get("status") == "completed"]
    if workspace:
        t.add_row("workspace", f"{len(pending)} active · {len(completed)} completed · {len(workspace)} total")

    console.print(Panel(t, title="Current state", border_style="cyan"))


def _print_tools(console: Console, tool_status_fn: ToolStatusFn) -> None:
    if tool_status_fn is None:
        console.print("[red]tool status not available.[/red]")
        return
    snap = tool_status_fn() or {}
    tools = snap.get("tools") or {}

    t = Table(show_header=True, box=None, padding=(0, 2))
    t.add_column("Tool", style="cyan")
    t.add_column("Category", style="dim")
    t.add_column("Available")
    t.add_column("Mode", style="dim")   # inline vs background
    t.add_column("Confirm?", style="dim")

    for name, info in tools.items():
        avail = "[green]yes[/green]" if info.get("available") else "[red]no[/red]"
        confirm = "[yellow]yes[/yellow]" if info.get("needs_confirmation") else "no"
        mode = "[magenta]background ⟳[/magenta]" if info.get("background") else "inline"
        t.add_row(name, info.get("category", ""), avail, mode, confirm)

    n_bg = sum(1 for i in tools.values() if i.get("background"))
    header = (
        f"Registered: {snap.get('registered', 0)} · "
        f"Available: {snap.get('available', 0)} · "
        f"Background: {n_bg}"
    )
    console.print(Panel(t, title=f"Tools ({header})", border_style="cyan"))


def _print_workspace(console: Console, inspector: InspectorFn) -> None:
    """Show AgenticWorkspace — active + completed background tasks."""
    if inspector is None:
        console.print("[red]workspace not available — no inspector wired.[/red]")
        return
    snap = inspector() or {}
    workspace = snap.get("workspace") or []
    bg_active = snap.get("background_tasks_active", 0)

    if not workspace:
        console.print(
            Panel(
                f"[dim]No workspace items. Active asyncio tasks: {bg_active}[/dim]",
                title="AgenticWorkspace",
                border_style="cyan",
            )
        )
        return

    t = Table(show_header=True, box=None, padding=(0, 2))
    t.add_column("Tool / Title", style="cyan")
    t.add_column("Status")
    t.add_column("Updated", style="dim")
    t.add_column("Result preview", style="dim")

    STATUS_COLOR = {
        "pending":     "dim",
        "in_progress": "yellow",
        "completed":   "green",
        "failed":      "red",
        "blocked":     "red",
        "handled":     "dim",
    }

    for item in workspace:
        status = item.get("status", "?")
        color = STATUS_COLOR.get(status, "white")
        updated = item.get("updated_at", "")[:19]
        desc = (item.get("description") or "")[:60]
        t.add_row(
            item.get("title", "?"),
            f"[{color}]{status}[/{color}]",
            updated,
            desc,
        )

    console.print(Panel(
        t,
        title=f"AgenticWorkspace ({len(workspace)} items · {bg_active} asyncio tasks active)",
        border_style="cyan",
    ))


def _print_memory(console: Console, inspector: InspectorFn) -> None:
    if inspector is None:
        console.print("[red]memory snapshot not available — no inspector wired.[/red]")
        return
    snap = inspector() or {}
    mem = snap.get("memory") or {}

    def _fmt_list(label: str, items: list) -> Text:
        head = Text(f"{label} ({len(items)})\n", style="bold")
        if not items:
            head.append("  (none)\n", style="dim")
            return head
        for m in items[:5]:
            ident = (
                m.get("person_name") or m.get("concept")
                or (m.get("content") or m.get("root_content") or "?")[:80]
            )
            head.append(f"  · {ident}\n")
        if len(items) > 5:
            head.append(f"  … +{len(items)-5} more\n", style="dim")
        return head

    parts = Text()
    parts.append_text(_fmt_list("must_know", mem.get("must_know") or []))
    parts.append("\n")
    parts.append_text(_fmt_list("context", mem.get("context") or []))
    parts.append("\n")
    parts.append_text(_fmt_list("associations", mem.get("associations") or []))

    console.print(Panel(parts, title="Memories surfaced last turn", border_style="cyan"))


def _handle_slash(
    command: str,
    console: Console,
    clear_history: ClearFn,
    inspector: InspectorFn,
    set_mode: SetModeFn,
    tool_status_fn: ToolStatusFn = None,
) -> bool:
    """Return True if the CLI should exit."""
    raw = command.strip()
    parts = raw.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        console.print("[dim]exiting.[/dim]")
        return True
    if cmd == "/clear":
        clear_history()
        console.print("[dim]in-session conversation cleared. (Memory graph untouched.)[/dim]")
        return False
    if cmd == "/help" or cmd == "/?":
        _print_help(console)
        return False
    if cmd == "/status":
        _print_status(console, inspector)
        return False
    if cmd == "/memory":
        _print_memory(console, inspector)
        return False
    if cmd == "/workspace":
        _print_workspace(console, inspector)
        return False
    if cmd == "/mode":
        if not arg:
            console.print("[yellow]usage: /mode <conversational|empathetic|focused|creative|auto>[/yellow]")
            return False
        if set_mode is None:
            console.print("[red]/mode not available — no set_mode wired.[/red]")
            return False
        try:
            if arg.lower() == "auto":
                set_mode("auto")
                console.print("[dim]mode: auto (controller-driven)[/dim]")
            else:
                set_mode(arg.lower())
                console.print(f"[dim]mode forced: {arg.lower()}[/dim]")
        except Exception as exc:
            console.print(f"[red]/mode failed: {exc}[/red]")
        return False

    if cmd == "/tools":
        _print_tools(console, tool_status_fn)
        return False

    # /reload is intentionally NOT handled here — it's async and is caught
    # directly in the run_cli loop before this function is called.

    console.print(f"[red]unknown command: {command}[/red]")
    _print_help(console)
    return False


# ============================================================================
# Reload helper (async — cannot live inside the sync _handle_slash)
# ============================================================================

async def _handle_reload(console: Console, reload_fn: ReloadFn) -> None:
    """Hot-reload all BRAIN modules in ~2-3s. Memory stack is untouched."""
    from rich.status import Status

    if reload_fn is None:
        console.print("[yellow]/reload not wired — pass reload_fn= to run_cli.[/yellow]")
        return

    try:
        with Status("[cyan]reloading brain…[/cyan]", console=console, spinner="dots"):
            await reload_fn()
        console.print(
            "[green]Reloaded.[/green] [dim]Memory, Neo4j, and ML models unchanged.[/dim]"
        )
    except Exception as exc:
        console.print(f"[red]Reload failed:[/red] {exc}")


# ============================================================================
# Public entry point
# ============================================================================

ProcessProactiveFn = Optional[Callable]  # (item) -> AsyncIterator[str]


async def run_cli(
    process: ProcessFn,
    clear_history: ClearFn,
    inspector: InspectorFn = None,
    set_mode: SetModeFn = None,
    greeting: Optional[str] = None,
    tool_status_fn: ToolStatusFn = None,
    tool_calls_fn: ToolCallsFn = None,
    reload_fn: ReloadFn = None,
    set_confirmation_fn: SetConfirmFn = None,
    get_proactive_fn=None,   # () -> list[WorkspaceItem] — completed background tasks
    process_proactive_fn: ProcessProactiveFn = None,  # (item) -> AsyncIterator[str]
) -> None:
    """
    Run the interactive SOFi terminal.

    Args:
        process:             async callable; given a user message, returns a token iterator.
        clear_history:       sync callable to wipe brain's in-memory short history.
        inspector:           optional sync callable returning a dict snapshot.
        set_mode:            optional sync callable to force a mode (or 'auto').
        greeting:            optional cold-start greeting line.
        tool_status_fn:      returns tool registry status for /tools command.
        tool_calls_fn:       returns list of tool calls from the last turn.
        reload_fn:           async callable that hot-reloads BRAIN code; returns new Brain.
        set_confirmation_fn: called once after PromptSession is ready; receives the
                             async confirmation callback so brain can wire it in.
    """
    console = Console()

    # ─── Banner ────────────────────────────────────────────────────────────
    tool_count = 0
    if tool_status_fn:
        try:
            ts = tool_status_fn()
            tool_count = ts.get("available", 0)
        except Exception:
            pass

    banner_text = (
        "[bold cyan]SOFi[/bold cyan] · personal AI companion\n"
        f"[dim]Type /help for commands. {tool_count} tools available. Ctrl-D or /exit to quit.[/dim]"
    )
    console.print(Panel(
        Text.from_markup(banner_text),
        border_style="cyan",
    ))

    line = greeting or _greeting_for_time()
    console.print(Panel(
        Markdown(line),
        title="SOFi",
        title_align="left",
        border_style="cyan",
    ))

    # ─── Input session ─────────────────────────────────────────────────────
    kb = KeyBindings()

    @kb.add("/")
    def _slash_key(event) -> None:
        """Insert '/' and immediately open the completion popup."""
        buf = event.app.current_buffer
        buf.insert_text("/")
        # Only trigger the dropdown when '/' is the very first character
        # so URLs and paths mid-sentence don't pop the menu.
        if buf.text == "/":
            buf.start_completion(select_first=False)

    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        multiline=False,
        key_bindings=kb,
        completer=SlashCommandCompleter(),
        complete_while_typing=True,
    )

    # ─── Confirmation handler ───────────────────────────────────────────────
    # Called by brain.py pre-flight before any dangerous tool executes.
    # Runs input() in a thread-pool executor so the event loop stays live.
    # Rich handles the print correctly even while a Live panel is active.

    async def _ask_confirmation(tool_name: str, question: str) -> bool:
        loop = asyncio.get_event_loop()
        console.print(f"\n[bold yellow]⚠[/bold yellow]  {question}")
        answer: str = await loop.run_in_executor(
            None,
            lambda: input("  y / n ▸ ").strip().lower(),
        )
        approved = answer in ("y", "yes")
        if approved:
            console.print("[dim]  confirmed.[/dim]")
        else:
            console.print("[dim]  declined — not executing.[/dim]")
        return approved

    if set_confirmation_fn is not None:
        set_confirmation_fn(_ask_confirmation)

    # ─── Main loop ─────────────────────────────────────────────────────────
    _last_proactive_ts: float = 0.0   # rate-limit: max 1 proactive notify / 30s
    _PROACTIVE_RATE_S: float = 30.0
    _PROACTIVE_POLL_S: float = 2.0    # check every 2s while prompt is idle
    # Items drained from brain's queue during mid-prompt polling that weren't
    # urgent enough to interrupt — held here and processed before the next prompt.
    _deferred_proactive: list = []

    while True:
        # ── Proactive items: checked BEFORE showing the prompt ──
        # Processes both items deferred from the last polling tick AND any new
        # items from brain's queue (all priorities — no active prompt to interrupt).
        if process_proactive_fn:
            try:
                pending = list(_deferred_proactive)
                _deferred_proactive.clear()
                if get_proactive_fn:
                    pending.extend(get_proactive_fn())
                for item in pending:
                    console.print(
                        f"\n[bold cyan]●[/bold cyan] "
                        f"[cyan]{item.title[:70]}[/cyan]"
                    )
                    await _render_stream(
                        console,
                        process_proactive_fn(item),
                        inspector=inspector,
                        tool_calls_fn=tool_calls_fn,
                    )
            except Exception:
                pass  # proactive check must never crash the main loop

        # ── Prompt with 2s URGENT polling ──
        # We run session.prompt_async() as an asyncio task and check for
        # URGENT proactive items every 2 seconds while it's idle.
        # URGENT items cancel the prompt, announce immediately, then re-prompt.
        user_input: Optional[str] = None
        try:
            with patch_stdout():
                _prompt_task = asyncio.ensure_future(
                    session.prompt_async("you ▸ ")
                )
                while True:
                    done, _ = await asyncio.wait(
                        [_prompt_task],
                        timeout=_PROACTIVE_POLL_S,
                    )
                    if done:
                        user_input = _prompt_task.result()
                        break

                    # 2s tick — check for URGENT proactive items
                    if get_proactive_fn and process_proactive_fn:
                        try:
                            all_items = get_proactive_fn()
                            urgent = [
                                i for i in all_items
                                if str(getattr(i, "notify_priority", "low")).lower() == "urgent"
                            ]
                            non_urgent = [
                                i for i in all_items
                                if str(getattr(i, "notify_priority", "low")).lower() != "urgent"
                            ]

                            # Non-urgent items: defer to the top of the next loop
                            # iteration (processed before the next prompt).
                            _deferred_proactive.extend(non_urgent)

                            now = time.perf_counter()
                            if urgent and (now - _last_proactive_ts) >= _PROACTIVE_RATE_S:
                                _last_proactive_ts = now
                                _prompt_task.cancel()
                                try:
                                    await _prompt_task
                                except (asyncio.CancelledError, Exception):
                                    pass
                                # Process one urgent item now; defer the rest.
                                console.print(
                                    f"\n[bold cyan]●[/bold cyan] "
                                    f"[cyan]{urgent[0].title[:70]}[/cyan]"
                                )
                                await _render_stream(
                                    console,
                                    process_proactive_fn(urgent[0]),
                                    inspector=inspector,
                                    tool_calls_fn=tool_calls_fn,
                                )
                                _deferred_proactive.extend(urgent[1:])
                                # Re-prompt after the proactive response
                                _prompt_task = asyncio.ensure_future(
                                    session.prompt_async("you ▸ ")
                                )
                        except Exception:
                            pass  # proactive polling must never crash the loop
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]exiting.[/dim]")
            return

        message = user_input.strip()
        if not message:
            continue

        if message.startswith("/"):
            # /reload is async — intercept before the sync handler.
            if message.strip().lower().split()[0] == "/reload":
                await _handle_reload(console, reload_fn)
                continue

            if _handle_slash(
                message, console, clear_history, inspector, set_mode,
                tool_status_fn=tool_status_fn,
            ):
                return
            continue

        try:
            await _render_stream(
                console,
                process(message),
                inspector=inspector,
                tool_calls_fn=tool_calls_fn,
            )
        except Exception as exc:
            console.print(f"[bold red]error:[/bold red] {type(exc).__name__}: {exc}")


def _build_status_line(inspector: InspectorFn) -> Optional[str]:
    """Compact 'mode=X · emotion=Y · intensity=Z' line for under each response."""
    if inspector is None:
        return None
    try:
        snap = inspector() or {}
        mode = snap.get("mode")
        ust = snap.get("user_state") or {}
        emo = ust.get("current_emotional_state")
        intensity = ust.get("emotional_intensity")
        parts = []
        if mode:
            parts.append(f"mode={mode}")
        if emo and emo != "neutral":
            parts.append(f"emotion={emo}")
        if intensity and float(intensity) > 0.05:
            parts.append(f"intensity={intensity}")
        return " · ".join(parts) if parts else None
    except Exception:
        return None
