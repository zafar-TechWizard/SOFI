"""
sofi.py — Terminal entry point for SOFi.

Run from the workspace root:
    python sofi.py

Phase 2 surface: persona + long-term memory wired. First boot takes ~30s
(Docker + Neo4j + model warmups); subsequent turns are warm.
"""

import asyncio
import logging
import signal
import sys
import time
import warnings
from pathlib import Path

# Suppress third-party noise before any imports that trigger them.
# NOTE: module= uses re.escape + \Z, so module="gliner" only matches the literal string "gliner",
#       NOT submodules like "gliner.data_processing.processor". Use message= filters instead.
warnings.filterwarnings("ignore", message="Sentence of length")          # GLiNER: truncation warning
warnings.filterwarnings("ignore", message="truncated to")                # GLiNER: secondary truncation variant
warnings.filterwarnings("ignore", message="renamed to")                  # old duckduckgo_search → ddgs
warnings.filterwarnings("ignore", message="non-text parts")              # Google SDK thought_signature
warnings.filterwarnings("ignore", message="thought_signature")           # Google SDK hidden thoughts
warnings.filterwarnings("ignore", category=ResourceWarning)              # unclosed SSL sockets from httpx/ddgs

from dotenv import load_dotenv

# Ensure workspace root is on sys.path so `import BRAIN...` and `import memory...` work
# regardless of how the script is launched.
_ROOT = Path(__file__).parent.absolute()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env BEFORE importing anything that reads GROQ_API_KEY at import time.
load_dotenv(_ROOT / ".env")

from rich.console import Console
from rich.status import Status

from BRAIN.brain import Brain
from BRAIN.ui.cli import run_cli


def _configure_logging(debug: bool = False) -> None:
    """
    Set up logging for the sofi.brain namespace.

    Logs go to BRAIN/memory/data/logs/brain_YYYY-MM-DD.log.
    Pass --debug on the CLI to also get DEBUG-level logs (very verbose).

    The memory system has its own separate observability (observer singleton),
    controlled by Brain(memory_log=True). These are different log sinks.
    """
    import datetime

    log_dir = _ROOT / "BRAIN" / "memory" / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().isoformat()
    log_file = log_dir / f"brain_{today}.log"

    level = logging.DEBUG if debug else logging.INFO

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    # Wire sofi.brain.* to this file handler
    logger = logging.getLogger("sofi")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False  # Don't bubble to root logger

    logger.info("=== SOFi brain log started | debug=%s | file=%s ===", debug, log_file)


async def main() -> None:
    debug_mode = "--debug" in sys.argv
    _configure_logging(debug=debug_mode)

    console = Console()

    # memory_log=True enables the memory system's own observability (observer singleton).
    # debug_mode also enables verbose brain orchestration logs via Python logging.
    brain = Brain(memory_log=debug_mode, memory_review=False)

    # Boot indicator — memory setup takes ~30s; without feedback the user
    # would see a frozen terminal.
    t0 = time.perf_counter()
    with Status("[cyan]starting SOFi…[/cyan]", console=console, spinner="dots") as status:
        def on_progress(stage: str) -> None:
            status.update(f"[cyan]{stage}[/cyan]")
        await brain.setup(on_progress=on_progress)
    boot_ms = (time.perf_counter() - t0) * 1000
    console.print(f"[dim]boot {boot_ms:.0f}ms[/dim]")

    # Mutable ref — /reload swaps _ref[0] to a new Brain without restarting
    # the memory stack (Neo4j, GLiNER, and cross-encoder stay loaded).
    _ref: list = [brain]

    async def _reload() -> None:
        new = await _ref[0].hot_reload()
        _ref[0] = new

    # Graceful shutdown on SIGINT/SIGTERM — ensures memory flush, consolidation,
    # and background task cleanup happen instead of hard-killing the process.
    _shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            sys.exit(1)
        _shutdown_requested = True
        console.print("\n[dim]shutting down gracefully…[/dim]")

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        await run_cli(
            process=lambda msg: _ref[0].process(msg),
            clear_history=lambda: _ref[0].clear_history(),
            inspector=lambda: _ref[0].inspect(),
            set_mode=lambda name: _ref[0].force_mode(name),
            tool_status_fn=lambda: _ref[0].tool_status(),
            tool_calls_fn=lambda: _ref[0].last_tool_calls,
            reload_fn=_reload,
            set_confirmation_fn=lambda fn: _ref[0].set_confirmation_handler(fn),
            get_proactive_fn=lambda: _ref[0].get_pending_proactive(),
        )
    finally:
        await _ref[0].shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
