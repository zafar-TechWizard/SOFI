"""
BRAIN/tools/fs_tools.py — Local filesystem tools for SOFi

- read_file      : Read any text file
- list_directory : Browse a directory
- search_files   : Find files by name pattern or content

All paths are resolved relative to the assistant workspace root if not absolute.
"""

import asyncio
import fnmatch
import logging
import os
from pathlib import Path

_log = logging.getLogger("sofi.brain.tools.fs")

_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()  # assistant/

_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB hard cap


# ── Read File ─────────────────────────────────────────────────────────────────

async def read_file(path: str, max_chars: int = 6000, offset_line: int = 0) -> str:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = _WORKSPACE_ROOT / p
        p = p.resolve()

        if not p.exists():
            return f"File not found: {path}"
        if not p.is_file():
            return f"Not a file: {path}"

        size = p.stat().st_size
        if size > _MAX_FILE_SIZE_BYTES:
            return f"File too large ({size // 1024}KB). Use search_files or specify a smaller range."

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None, lambda: p.read_text(encoding="utf-8", errors="replace")
        )

        lines = content.splitlines()
        total_lines = len(lines)

        if offset_line > 0:
            lines = lines[offset_line:]
            content = "\n".join(lines)

        if len(content) > max_chars:
            content = content[:max_chars]
            return (
                f"File: {p}\n"
                f"Lines: {total_lines} total"
                + (f" (showing from line {offset_line})" if offset_line else "")
                + f", first {max_chars} chars shown\n\n{content}\n\n[... truncated]"
            )

        return (
            f"File: {p}\n"
            f"Lines: {total_lines}"
            + (f" (from line {offset_line})" if offset_line else "")
            + f"\n\n{content}"
        )

    except Exception as exc:
        _log.error("read_file error path=%s: %s", path, exc)
        return f"Error reading file: {exc}"


# ── List Directory ────────────────────────────────────────────────────────────

async def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = _WORKSPACE_ROOT / p
        p = p.resolve()

        if not p.exists():
            return f"Directory not found: {path}"
        if not p.is_file():
            pass  # it's a dir — continue
        if p.is_file():
            return f"That's a file, not a directory: {path}"

        loop = asyncio.get_event_loop()

        def _list():
            items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            if not show_hidden:
                items = [i for i in items if not i.name.startswith(".")]
            return items

        items = await loop.run_in_executor(None, _list)

        if not items:
            return f"Empty directory: {p}"

        dirs = [i for i in items if i.is_dir()]
        files = [i for i in items if i.is_file()]

        lines = [f"Directory: {p}\n"]
        for d in dirs:
            lines.append(f"  {d.name}/")
        for f in files:
            try:
                size = f.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
            except OSError:
                size_str = "?"
            lines.append(f"  {f.name}  ({size_str})")

        lines.append(f"\n{len(dirs)} dir(s), {len(files)} file(s)")
        return "\n".join(lines)

    except Exception as exc:
        _log.error("list_directory error path=%s: %s", path, exc)
        return f"Error listing directory: {exc}"


# ── Search Files ──────────────────────────────────────────────────────────────

_SKIP_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "venv", "dist", "build", ".mypy_cache"}

async def search_files(
    directory: str = ".",
    pattern: str = "*",
    content_search: str = "",
    max_results: int = 20,
) -> str:
    try:
        p = Path(directory).expanduser()
        if not p.is_absolute():
            p = _WORKSPACE_ROOT / p
        p = p.resolve()

        if not p.exists():
            return f"Directory not found: {directory}"

        loop = asyncio.get_event_loop()

        def _search():
            found = []
            for root, dirs, files in os.walk(p):
                # Prune noisy directories in-place so os.walk skips them
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
                for fname in files:
                    if not fnmatch.fnmatch(fname, pattern):
                        continue
                    fpath = Path(root) / fname
                    if content_search:
                        try:
                            text = fpath.read_text(encoding="utf-8", errors="ignore")
                            if content_search.lower() not in text.lower():
                                continue
                        except OSError:
                            continue
                    found.append(str(fpath))
                    if len(found) >= max_results:
                        return found, True
            return found, False

        matches, capped = await loop.run_in_executor(None, _search)

        if not matches:
            desc = f"pattern '{pattern}'"
            if content_search:
                desc += f" containing '{content_search}'"
            return f"No files found in {p} matching {desc}"

        lines = [f"Found {len(matches)} file(s) in {p}:"]
        for m in matches:
            lines.append(f"  {m}")
        if capped:
            lines.append(f"\n(Capped at {max_results} results — narrow your search for more precision)")

        return "\n".join(lines)

    except Exception as exc:
        _log.error("search_files error dir=%s: %s", directory, exc)
        return f"Error searching files: {exc}"


# ── Registration ──────────────────────────────────────────────────────────────

def register_fs_tools(registry) -> None:
    from BRAIN.tools.registry import ToolEntry

    registry.register(ToolEntry(
        name="read_file",
        description=(
            "Read the contents of a file on the local filesystem. "
            "Supports any text file (code, markdown, JSON, logs, etc). "
            "Relative paths are resolved from the assistant workspace root. "
            "Use offset_line to read from a specific line onwards."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to workspace root, or absolute)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return (default 6000)",
                    "default": 6000,
                },
                "offset_line": {
                    "type": "integer",
                    "description": "Start reading from this line number (0 = beginning)",
                    "default": 0,
                },
            },
            "required": ["path"],
        },
        handler=read_file,
        category="filesystem",
        capability_name="read_file",
        capability_description="Read files from the local filesystem.",
        capability_refusal="I can't access the filesystem right now.",
    ))

    registry.register(ToolEntry(
        name="list_directory",
        description=(
            "List files and subdirectories in a given directory. "
            "Shows file sizes. Relative paths resolve from the workspace root. "
            "Use '.' or leave blank for the workspace root."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: workspace root)",
                    "default": ".",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files/dirs (starting with '.')",
                    "default": False,
                },
            },
            "required": [],
        },
        handler=list_directory,
        category="filesystem",
        capability_name="list_directory",
        capability_description="Browse the local filesystem directory structure.",
        capability_refusal="I can't access the filesystem right now.",
    ))

    registry.register(ToolEntry(
        name="search_files",
        description=(
            "Recursively search for files by name pattern and/or content. "
            "pattern uses glob syntax: '*.py', '*.md', 'config*', etc. "
            "content_search filters to only files containing that text. "
            "Skips __pycache__, .git, node_modules automatically."
        ),
        schema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Root directory to search (default: workspace root)",
                    "default": ".",
                },
                "pattern": {
                    "type": "string",
                    "description": "Filename glob pattern e.g. '*.py', '*.json', 'brain*'",
                    "default": "*",
                },
                "content_search": {
                    "type": "string",
                    "description": "Optional: only return files containing this text",
                    "default": "",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max files to return (default 20)",
                    "default": 20,
                },
            },
            "required": [],
        },
        handler=search_files,
        category="filesystem",
        capability_name="search_files",
        capability_description="Search for files by name or content on the local filesystem.",
        capability_refusal="I can't search files right now.",
    ))
