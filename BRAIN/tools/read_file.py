"""
BRAIN/tools/read_file.py — File reading tool.

Matches Claude Code's read quality:
  - Line-numbered output (cat -n style) for precise patch targeting
  - Windowed reads: offset + limit — never dumps entire large files
  - Binary detection: reports type/size instead of dumping garbage
  - Multi-encoding: UTF-16 BOM → UTF-8 BOM → UTF-8 → CP-1252 → Latin-1
  - Two read paths: full-read for files <5 MB, readline-stream for ≥5 MB
  - All I/O in asyncio.to_thread — never blocks the event loop
  - Helpful "file not found" errors with sibling listing
  - Symlink transparency: notes the real path in the header
"""

import asyncio
import logging
import mimetypes
import os
from pathlib import Path

from BRAIN.tools.registry import ToolEntry

_log = logging.getLogger("sofi.brain.tools.read_file")

# ── Tuning constants ───────────────────────────────────────────────────────────

DEFAULT_LIMIT       = 200               # lines per call if caller doesn't specify
MAX_LIMIT           = 2_000            # hard ceiling regardless of caller's request
FULL_READ_THRESHOLD = 5 * 1024 * 1024  # 5 MB: below this, load whole file into RAM
MAX_LINE_CHARS      = 2_000            # chars per line before truncation with notice
BINARY_SNIFF_BYTES  = 8_192            # bytes examined for null-byte binary detection


# ── Pure helpers (no I/O) ──────────────────────────────────────────────────────

def _is_binary(sample: bytes) -> bool:
    """Null-byte heuristic: any null in first 8 KB → binary."""
    return b"\x00" in sample


def _decode(raw: bytes) -> tuple[str, str]:
    """
    Decode raw bytes to (text, encoding_label). Never raises.

    Tries in order: UTF-16 BOM → UTF-8 BOM (stripped) → UTF-8 →
    CP-1252 (Windows Western) → Latin-1 (last resort, never fails).
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16"), "utf-16"
        except Exception:
            pass
    if raw[:3] == b"\xef\xbb\xbf":
        try:
            return raw[3:].decode("utf-8"), "utf-8"
        except Exception:
            pass
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except (UnicodeDecodeError, LookupError):
        pass
    return raw.decode("latin-1"), "latin-1"


def _human_size(n: int) -> str:
    if n < 1_024:
        return f"{n} B"
    if n < 1_024 ** 2:
        return f"{n / 1_024:.1f} KB"
    return f"{n / 1_024 ** 2:.1f} MB"


def _mime(path: Path) -> str:
    m, _ = mimetypes.guess_type(str(path))
    return m or "application/octet-stream"


def _numbered(lines: list[str], first_lineno: int) -> str:
    """
    Format lines with right-aligned 1-based line numbers, tab-separated (cat -n style).

    Numbers are padded to the width of the last line number so columns
    stay aligned across windowed reads. Lines longer than MAX_LINE_CHARS
    are truncated with a suffix showing how many characters were omitted.
    """
    last_no = first_lineno + len(lines) - 1
    width   = len(str(last_no))
    out     = []
    for i, line in enumerate(lines):
        no = first_lineno + i
        if len(line) > MAX_LINE_CHARS:
            dropped = len(line) - MAX_LINE_CHARS
            line = line[:MAX_LINE_CHARS] + f" [+{dropped} chars omitted]"
        out.append(f"{no:{width}d}\t{line}")
    return "\n".join(out)


# ── Sync I/O helpers (always called via asyncio.to_thread) ────────────────────

def _sniff(path: Path) -> bytes:
    """Read first BINARY_SNIFF_BYTES for binary detection."""
    with open(path, "rb") as fh:
        return fh.read(BINARY_SNIFF_BYTES)


def _read_full(path: Path, offset: int, limit: int) -> tuple[list[str], int, str]:
    """
    Load entire file into RAM, decode, split lines, return the window.
    Used for files under FULL_READ_THRESHOLD.
    Returns (window_lines, total_lines, encoding_label).
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    text, encoding = _decode(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    all_lines = text.split("\n")
    # Don't show a spurious empty line for files that end with a newline
    if all_lines and all_lines[-1] == "":
        all_lines = all_lines[:-1]

    total  = len(all_lines)
    window = all_lines[offset : offset + limit]
    return window, total, encoding


def _read_stream(path: Path, offset: int, limit: int) -> tuple[list[str], int, str]:
    """
    Stream file line by line — never loads whole file into RAM.
    Used for files at or above FULL_READ_THRESHOLD.
    Returns (window_lines, total_lines, encoding_label).
    """
    # Peek at the first 4 bytes to choose encoding without consuming the stream.
    with open(path, "rb") as fh:
        head = fh.read(4)

    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16"
    elif head[:3] == b"\xef\xbb\xbf":
        enc = "utf-8-sig"   # Python's built-in BOM-stripping UTF-8 codec
    else:
        enc = "utf-8"

    def _iter(open_enc: str) -> tuple[list[str], int]:
        window_: list[str] = []
        total_ = 0
        with open(path, "r", encoding=open_enc, errors="replace", newline="") as fh:
            for i, raw_line in enumerate(fh):
                total_ += 1
                if i >= offset and len(window_) < limit:
                    window_.append(raw_line.rstrip("\r\n"))
        return window_, total_

    try:
        window, total = _iter(enc)
    except (UnicodeDecodeError, LookupError):
        enc = "latin-1"
        window, total = _iter(enc)

    label = "utf-8" if enc == "utf-8-sig" else enc
    return window, total, label


def _siblings(resolved: Path, limit: int = 8) -> list[str]:
    """List visible entries in the parent directory — for helpful 'not found' errors."""
    try:
        names = sorted(
            e.name for e in resolved.parent.iterdir()
            if not e.name.startswith(".")
        )
        had_more = len(names) > limit
        result   = names[:limit]
        if had_more:
            result.append("…")
        return result
    except Exception:
        return []


# ── Async entry point ──────────────────────────────────────────────────────────

async def read_file(path: str, offset: int = 0, limit: int = DEFAULT_LIMIT) -> str:
    """
    Read a file and return its contents with 1-based line numbers.

    This is the public handler registered in the tool registry.
    All blocking I/O runs in asyncio.to_thread so the event loop is never stalled.
    """

    # ── Normalise parameters ──────────────────────────────────────────────────
    limit  = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))

    target = Path(path)
    try:
        resolved = target.resolve()
    except OSError:
        # Symlink loop or path too long — degrade gracefully
        resolved = target.absolute()

    # ── Existence + type checks ───────────────────────────────────────────────
    if not resolved.exists():
        hint_lines = _siblings(resolved)
        hint = ""
        if hint_lines and resolved.parent.exists():
            hint = f"\n\nContents of {resolved.parent}:\n  " + "\n  ".join(hint_lines)
        return f"File not found: {resolved}{hint}"

    if resolved.is_dir():
        return (
            f"'{resolved}' is a directory, not a file.\n"
            f"Use list_directory(path='{path}') to explore its contents."
        )

    if not resolved.is_file():
        return f"'{resolved}' is not a regular file (socket, pipe, or device node)."

    # ── Permission ────────────────────────────────────────────────────────────
    if not os.access(resolved, os.R_OK):
        return f"Permission denied: cannot read '{resolved}'."

    # ── File metadata ─────────────────────────────────────────────────────────
    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        return f"Cannot stat '{resolved}': {exc}"

    # Show the real path when a symlink was followed
    symlink_note = f" → {resolved}" if target.is_symlink() else ""

    # ── Empty file ────────────────────────────────────────────────────────────
    if size_bytes == 0:
        return (
            f"{target}{symlink_note}\n"
            f"(0 lines · 0 B)\n\n"
            f"(empty file)"
        )

    # ── Binary detection ──────────────────────────────────────────────────────
    try:
        sample = await asyncio.to_thread(_sniff, resolved)
    except OSError as exc:
        return f"Error reading '{resolved}': {exc}"

    if _is_binary(sample):
        mime = _mime(resolved)
        return (
            f"{target}{symlink_note}\n"
            f"Binary · {mime} · {_human_size(size_bytes)}\n\n"
            f"File contains binary data — cannot display as text.\n"
            f"Suggestions:\n"
            f"  run_command('file \"{resolved}\"')           — identify exact file type\n"
            f"  run_command('xxd \"{resolved}\" | head -20') — hex dump first 20 lines"
        )

    # ── Read (two paths based on file size) ───────────────────────────────────
    try:
        if size_bytes < FULL_READ_THRESHOLD:
            window, total_lines, encoding = await asyncio.to_thread(
                _read_full, resolved, offset, limit
            )
        else:
            _log.debug("read_file | streaming | path=%s size=%s", resolved, _human_size(size_bytes))
            window, total_lines, encoding = await asyncio.to_thread(
                _read_stream, resolved, offset, limit
            )
    except PermissionError:
        return f"Permission denied reading '{resolved}'."
    except OSError as exc:
        return f"I/O error reading '{resolved}': {exc}"

    # ── Offset validation ─────────────────────────────────────────────────────
    if total_lines == 0:
        return (
            f"{target}{symlink_note}\n"
            f"({_human_size(size_bytes)} · {encoding})\n\n"
            f"(no lines — file may contain only a newline or non-line content)"
        )

    if offset >= total_lines:
        return (
            f"{target}{symlink_note}\n"
            f"({total_lines} line{'s' if total_lines != 1 else ''} total "
            f"· {_human_size(size_bytes)} · {encoding})\n\n"
            f"offset={offset} is past the end of the file. "
            f"Valid offsets: 0–{total_lines - 1}."
        )

    if not window:
        return (
            f"{target}{symlink_note}\n"
            f"No lines in range (offset={offset}, limit={limit}, total={total_lines})."
        )

    # ── Format output ─────────────────────────────────────────────────────────
    first_no  = offset + 1            # 1-based line number of first returned line
    last_no   = offset + len(window)
    truncated = (offset + limit) < total_lines

    # Compact range description for the header
    if total_lines == 1:
        range_str = "1 line"
    elif first_no == last_no:
        range_str = f"line {first_no} of {total_lines}"
    else:
        range_str = f"lines {first_no}–{last_no} of {total_lines}"

    header = (
        f"{target}{symlink_note}\n"
        f"({range_str} · {_human_size(size_bytes)} · {encoding})"
    )
    body   = _numbered(window, first_no)

    parts = [header, "", body]

    if truncated:
        remaining = total_lines - last_no
        parts.append(
            f"\n[{len(window)} of {total_lines} lines shown — "
            f"{remaining} more. "
            f"Continue: offset={last_no}, limit={limit}]"
        )
    elif offset > 0:
        # Reading a window that reached the very end of the file
        parts.append(f"\n[End of file — {total_lines} lines total]")

    return "\n".join(parts)


# ── Registration ───────────────────────────────────────────────────────────────

def register(registry) -> None:
    registry.register(ToolEntry(
        name="read_file",
        description=(
            "Read a file and return its contents with 1-based line numbers.\n\n"
            "WINDOWING — For large files, read in pages using offset+limit. "
            "Each response shows how many lines remain and a 'Continue: offset=N, limit=M' "
            "hint — pass that offset on the next call to keep reading from where you stopped.\n\n"
            "OUTPUT FORMAT — Each line is formatted as '<lineno>\\t<content>'. "
            "Line numbers are right-aligned for visual consistency across pages. "
            "This is the format patch_file expects when targeting specific lines.\n\n"
            "HANDLES — Source code, configs, logs, Markdown, plain text, any UTF-8/UTF-16/Latin-1 file. "
            "Detects binary files (images, executables, archives) by null-byte scan and reports "
            "the file type and size instead of dumping garbage. "
            "Encoding falls back gracefully: UTF-8 → CP-1252 → Latin-1.\n\n"
            "EXAMPLES\n"
            "  read_file('main.py')                        — first 200 lines\n"
            "  read_file('main.py', offset=49, limit=30)  — lines 50–79\n"
            "  read_file('app.log', offset=200, limit=200) — lines 201–400 (next page)"
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute or relative path to the file. "
                        "Relative paths resolve from the current working directory."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "0-based index of the first line to return (default 0 = start of file). "
                        "To read lines 50–79, set offset=49. "
                        "Use the 'Continue: offset=N' value from a prior read to page forward."
                    ),
                    "default": 0,
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of lines to return (default {DEFAULT_LIMIT}, max {MAX_LIMIT}). "
                        "Increase for dense files like minified JS or data files. "
                        "The response always shows how many lines remain."
                    ),
                    "default": DEFAULT_LIMIT,
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                },
            },
            "required": ["path"],
        },
        handler=read_file,
        category="filesystem",
        timeout=15.0,
        capability_name="read_file",
        capability_description=(
            "Read any file from disk with windowed, line-numbered output. "
            "Handles large files (streaming), binary files (detection), and encoding issues (fallback)."
        ),
        capability_refusal="I can't read files right now.",
    ))
