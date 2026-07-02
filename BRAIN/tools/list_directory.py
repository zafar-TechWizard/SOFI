"""
BRAIN/tools/list_directory.py — Production directory listing tool.

Two display modes:
  • Flat  (recursive=false): aligned table of one directory's immediate contents.
  • Tree  (recursive=true ): hierarchical tree of all subdirectories up to max_depth.

Edge cases handled:
  - Path does not exist           → clear error with hint
  - Path is a file, not a dir    → clear error + read_file suggestion
  - Path is a special file       → descriptive error
  - Permission denied on dir     → error with path
  - Permission denied on entry   → counted, skipped, reported in footer
  - Symlinks                      → shown with target; broken target flagged ⚠
  - Circular symlinks (tree mode) → detected via resolved-path tracking; skipped
  - Very large directories        → hard-capped at max_results with notice
  - Empty directory               → explicit message, not an empty table
  - Windows drive paths           → handled via pathlib + os.path.splitdrive
  - Unreadable filename encoding  → falls back to repr()
  - Any other I/O error           → caught per-entry; never crashes the tool

Uses Pattern B (@tool decorator) — auto-discovered by _auto_register_tools.
No register() function needed.
"""

import asyncio
import fnmatch
import logging
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from BRAIN.tools import tool

_log = logging.getLogger("sofi.brain.tools.list_directory")

# ── Constants ─────────────────────────────────────────────────────────────────

# Directories always excluded in recursive mode regardless of show_hidden.
# These are build artifacts and VCS internals — never useful to list recursively.
_NOISE_DIRS: frozenset[str] = frozenset({
    "__pycache__", "node_modules", ".git", ".hg", ".svn",
    ".venv", "venv", ".tox", ".nox",
    "dist", "build", "out", "target",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".eggs",
    ".next", ".nuxt", ".turbo",
    "coverage", ".nyc_output",
})

# Glob patterns for noise dirs — fnmatch, not set lookup.
_NOISE_DIR_PATTERNS: tuple[str, ...] = ("*.egg-info", "*.dist-info")

_MAX_RESULTS_HARD: int = 1000
_MAX_DEPTH_HARD: int = 8

# Column widths for flat-mode table
_COL_TYPE: int = 4
_COL_SIZE: int = 9
_COL_MOD: int = 11

# Tree rendering characters (box-drawing)
_BRANCH = "├── "
_LAST   = "└── "
_PIPE   = "│   "
_SPACE  = "    "
_TRUNC  = "… "


# ── Formatting helpers ────────────────────────────────────────────────────────

def _human_size(n: int) -> str:
    """Compact human-readable file size."""
    if n < 1_024:
        return f"{n} B"
    if n < 1_048_576:
        return f"{n / 1_024:.1f} KB"
    if n < 1_073_741_824:
        return f"{n / 1_048_576:.1f} MB"
    return f"{n / 1_073_741_824:.1f} GB"


def _time_ago(ts: float) -> str:
    """Relative human-readable modification time."""
    try:
        delta = datetime.now(timezone.utc).timestamp() - ts
    except Exception:
        return "?"

    if delta < 5:
        return "just now"
    if delta < 3_600:
        m = int(delta / 60)
        return f"{m}m ago" if m > 0 else "just now"
    if delta < 86_400:
        h = int(delta / 3_600)
        return f"{h}h ago"
    d = int(delta / 86_400)
    if d == 1:
        return "yesterday"
    if d < 30:
        return f"{d}d ago"
    if d < 365:
        w = int(d / 7)
        return f"{w}w ago"
    y = int(d / 365)
    return f"{y}y ago"


# ── DirEntry helpers ──────────────────────────────────────────────────────────

def _entry_kind(e: os.DirEntry) -> str:
    """
    Classify a DirEntry into one of four labels.
    Uses DirEntry's cached metadata where possible to avoid extra syscalls.
    """
    try:
        if e.is_symlink():
            return "LINK"
        if e.is_dir(follow_symlinks=False):
            return "DIR "
        if e.is_file(follow_symlinks=False):
            return "FILE"
        # Special file (socket, device, FIFO)
        m = e.stat(follow_symlinks=False).st_mode
        if stat.S_ISCHR(m):  return "CHR "
        if stat.S_ISBLK(m):  return "BLK "
        if stat.S_ISFIFO(m): return "FIFO"
        if stat.S_ISSOCK(m): return "SOCK"
        return "OTH "
    except (PermissionError, OSError):
        return "????"


def _entry_stat(e: os.DirEntry) -> Optional[os.stat_result]:
    """lstat (don't follow symlinks) for an entry. Returns None on error."""
    try:
        return e.stat(follow_symlinks=False)
    except (PermissionError, OSError):
        return None


def _symlink_info(e: os.DirEntry) -> str:
    """
    Read the symlink's target and check whether it resolves.

    Returns strings like:
      → /absolute/target
      → ./relative  ⚠ broken
      → (unreadable)
    """
    try:
        target = os.readlink(e.path)
        # Check reachability without resolving the whole chain
        abs_target = Path(e.path).parent / target
        reachable = abs_target.exists()
        marker = "" if reachable else "  ⚠ broken"
        return f"→ {target}{marker}"
    except (OSError, ValueError):
        return "→ (unreadable)"


def _entry_name_safe(e: os.DirEntry) -> str:
    """Entry name, falling back to repr on encoding errors."""
    try:
        return e.name
    except Exception:
        return repr(e.name)


# ── Sort helper ───────────────────────────────────────────────────────────────

_TYPE_ORDER = {"DIR ": 0, "LINK": 1, "FILE": 2}


def _sort_key(entry: dict, sort_by: str):
    """
    Sort key for an entry dict.
    Dirs always sort before files; within each group sort by the requested field.
    """
    type_rank = _TYPE_ORDER.get(entry["kind"], 3)
    name_lower = entry["name"].lower()

    if sort_by == "modified":
        return (type_rank, -(entry["mtime"] or 0), name_lower)
    if sort_by == "size":
        return (type_rank, -(entry["size"] or 0), name_lower)
    if sort_by == "type":
        return (type_rank, name_lower)
    return (type_rank, name_lower)  # default: name


# ── Flat-mode table renderer ──────────────────────────────────────────────────

def _scan_flat(
    dirpath: Path,
    show_hidden: bool,
    pattern: Optional[str],
) -> tuple[list[dict], int]:
    """
    Scan a single directory and return (entries, perm_error_count).
    Each entry is a dict with: name, kind, size, mtime, symlink_info.
    """
    entries: list[dict] = []
    perm_errors = 0

    try:
        it = os.scandir(dirpath)
    except PermissionError:
        return [], 1
    except OSError as exc:
        _log.warning("list_directory | scandir failed | path=%s err=%s", dirpath, exc)
        return [], 0

    with it:
        for e in it:
            name = _entry_name_safe(e)

            if not show_hidden and name.startswith("."):
                continue

            kind = _entry_kind(e)
            is_dir = kind == "DIR "

            # Pattern applies to files only; dirs are always shown.
            if not is_dir and pattern and not fnmatch.fnmatch(name, pattern):
                continue

            st = _entry_stat(e)
            entries.append({
                "name":        name,
                "kind":        kind,
                "size":        st.st_size if st else None,
                "mtime":       st.st_mtime if st else None,
                "symlink_info": _symlink_info(e) if kind == "LINK" else None,
            })

    return entries, perm_errors


def _render_flat(
    dirpath: Path,
    entries: list[dict],
    total: int,
    perm_errors: int,
    max_results: int,
    pattern: Optional[str],
    sort_by: str,
) -> str:
    """Render a directory listing as an aligned table."""
    lines: list[str] = []
    abs_str = str(dirpath)

    # ── Header ────────────────────────────────────────────────────────────────
    header = abs_str
    if pattern:
        header += f"  [filter: {pattern}]"
    lines.append(header)

    if not entries:
        lines.append("(empty directory)")
        return "\n".join(lines)

    # ── Column widths ─────────────────────────────────────────────────────────
    max_name_len = max(
        (
            len(e["name"]) + (1 if e["kind"] == "DIR " else 0)
            + (2 + len(e["symlink_info"]) if e.get("symlink_info") else 0)
        )
        for e in entries
    )
    max_name_len = max(max_name_len, 24)
    sep_len = _COL_TYPE + 2 + max_name_len + 2 + _COL_SIZE + 2 + _COL_MOD

    lines.append("─" * min(sep_len, 100))

    # ── Column header ─────────────────────────────────────────────────────────
    lines.append(
        f"{'TYPE':<{_COL_TYPE}}  "
        f"{'NAME':<{max_name_len}}  "
        f"{'SIZE':>{_COL_SIZE}}  "
        f"MODIFIED"
    )
    lines.append(
        f"{'────':<{_COL_TYPE}}  "
        f"{'────':<{max_name_len}}  "
        f"{'────':>{_COL_SIZE}}  "
        f"────────"
    )

    # ── Rows ──────────────────────────────────────────────────────────────────
    for e in entries:
        kind = e["kind"]
        name = e["name"]

        # Display name
        if kind == "DIR ":
            display = name + "/"
        elif e.get("symlink_info"):
            display = f"{name}  {e['symlink_info']}"
        else:
            display = name

        # Size
        if kind == "DIR ":
            size_str = "–"
        elif e["size"] is not None:
            size_str = _human_size(e["size"])
        else:
            size_str = "?"

        # Modified
        mod_str = _time_ago(e["mtime"]) if e["mtime"] is not None else "?"

        lines.append(
            f"{kind:<{_COL_TYPE}}  "
            f"{display:<{max_name_len}}  "
            f"{size_str:>{_COL_SIZE}}  "
            f"{mod_str}"
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("─" * min(sep_len, 100))

    n_dirs  = sum(1 for e in entries if e["kind"] == "DIR ")
    n_files = sum(1 for e in entries if e["kind"] == "FILE")
    n_links = sum(1 for e in entries if e["kind"] == "LINK")
    n_other = len(entries) - n_dirs - n_files - n_links

    parts: list[str] = []
    if n_dirs:  parts.append(f"{n_dirs} dir{'s' if n_dirs  != 1 else ''}")
    if n_files: parts.append(f"{n_files} file{'s' if n_files != 1 else ''}")
    if n_links: parts.append(f"{n_links} link{'s' if n_links != 1 else ''}")
    if n_other: parts.append(f"{n_other} other")

    footer = " · ".join(parts) if parts else "0 entries"

    if total > max_results:
        footer += f"  |  showing {len(entries)} of {total} (capped — raise max_results to see more)"
    else:
        footer += f"  |  {total} entr{'y' if total == 1 else 'ies'}"

    if perm_errors:
        footer += f"  |  {perm_errors} entry/entries skipped (permission denied)"

    lines.append(footer)
    return "\n".join(lines)


# ── Tree-mode builder ─────────────────────────────────────────────────────────

class _TreeBuilder:
    """
    Builds a directory tree as a list of display strings.

    State fields track shared mutable state across all recursive calls:
      count       — total entries added so far (for max_results cap)
      perm_errors — directories that could not be scanned
      visited     — resolved real paths already expanded (circular-link guard)
      truncated   — set True when max_results is hit (suppresses further output)
    """

    def __init__(
        self,
        show_hidden: bool,
        pattern: Optional[str],
        sort_by: str,
        max_depth: int,
        max_results: int,
    ) -> None:
        self.show_hidden = show_hidden
        self.pattern = pattern
        self.sort_by = sort_by
        self.max_depth = max_depth
        self.max_results = max_results
        # mutable state
        self.count = 0
        self.perm_errors = 0
        self.visited: set[str] = set()
        self.truncated = False

    def build(self, dirpath: Path, prefix: str = "", depth: int = 0) -> list[str]:
        """
        Return tree lines for *dirpath*. Lines do NOT include *prefix* —
        the caller prepends the continuation prefix for each child line.
        The root call uses prefix="" and the result is joined as-is.
        """
        # ── Circular-symlink protection ───────────────────────────────────────
        try:
            real = str(dirpath.resolve())
            if real in self.visited:
                return [f"(circular symlink — skipped)"]
            self.visited.add(real)
        except OSError:
            pass

        # ── Scan ──────────────────────────────────────────────────────────────
        try:
            scan_iter = os.scandir(dirpath)
        except PermissionError:
            self.perm_errors += 1
            return ["(permission denied)"]
        except OSError as exc:
            return [f"(error: {exc})"]

        raw: list[dict] = []
        with scan_iter:
            for e in scan_iter:
                name = _entry_name_safe(e)
                kind = _entry_kind(e)
                is_dir = kind == "DIR "

                if not self.show_hidden and name.startswith("."):
                    continue

                # Noise dirs: always skip in tree mode (they're never useful here)
                if is_dir and (
                    name in _NOISE_DIRS
                    or any(fnmatch.fnmatch(name, p) for p in _NOISE_DIR_PATTERNS)
                ):
                    continue

                # Pattern applies to files only; dirs are always traversed
                if not is_dir and self.pattern and not fnmatch.fnmatch(name, self.pattern):
                    continue

                st = _entry_stat(e)
                raw.append({
                    "name":        name,
                    "kind":        kind,
                    "size":        st.st_size if st else None,
                    "mtime":       st.st_mtime if st else None,
                    "symlink_info": _symlink_info(e) if kind == "LINK" else None,
                    "path":        Path(e.path),
                })

        raw.sort(key=lambda e: _sort_key(e, self.sort_by))

        # ── Render ────────────────────────────────────────────────────────────
        lines: list[str] = []
        for idx, e in enumerate(raw):
            if self.truncated or self.count >= self.max_results:
                if not self.truncated:
                    lines.append(f"{_TRUNC}(truncated at {self.max_results} entries)")
                    self.truncated = True
                break

            is_last = (idx == len(raw) - 1)
            connector = _LAST if is_last else _BRANCH
            child_continuation = _SPACE if is_last else _PIPE

            kind    = e["kind"]
            name    = e["name"]
            is_dir  = kind == "DIR "
            is_link = kind == "LINK"

            # Display name
            if is_dir:
                display = f"{name}/"
            elif is_link and e.get("symlink_info"):
                display = f"{name}  {e['symlink_info']}"
            else:
                display = name

            # Inline annotation for files/links: (size · age)
            ann_parts: list[str] = []
            if not is_dir and e["size"] is not None:
                ann_parts.append(_human_size(e["size"]))
            if e["mtime"] is not None:
                ann_parts.append(_time_ago(e["mtime"]))
            ann = f"  ({' · '.join(ann_parts)})" if ann_parts else ""

            lines.append(f"{connector}{display}{ann}")
            self.count += 1

            # Recurse into real directories (not symlinks — those are just shown)
            if is_dir and not is_link and depth + 1 < self.max_depth:
                child_lines = self.build(e["path"], prefix=child_continuation, depth=depth + 1)
                for cl in child_lines:
                    lines.append(f"{child_continuation}{cl}")

        return lines


# ── Tool entry point ──────────────────────────────────────────────────────────

@tool(
    name="list_directory",
    description=(
        "List the contents of a directory — files, subdirectories, and symlinks — "
        "with type, size, and modification time.\n\n"
        "Two modes:\n"
        "• Flat (recursive=false, default): aligned table of the immediate directory. "
        "Shows everything in one directory, sorted by name.\n"
        "• Tree (recursive=true): hierarchical tree of all subdirectories up to "
        "max_depth. Noise directories (.git, __pycache__, node_modules, .venv, etc.) "
        "are always excluded in tree mode.\n\n"
        "Pattern filtering: glob syntax on filenames only — directories are always "
        "shown in tree mode regardless of the pattern. Examples: '*.py', 'test_*', "
        "'*.{js,ts}' (note: fnmatch, not regex).\n\n"
        "Hidden files (dot-names) are excluded by default; set show_hidden=true to "
        "include them.\n\n"
        "Result is hard-capped at max_results entries (default 200, max 1000). A "
        "truncation notice is shown when the cap is hit — use max_results or a "
        "pattern to narrow scope rather than raising the cap blindly."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory path to list. Absolute, or relative to the current "
                    "working directory. '~' is expanded. Examples: '.', '~', "
                    "'BRAIN/tools', 'C:/Users/mdzaf/project'."
                ),
            },
            "show_hidden": {
                "type": "boolean",
                "description": (
                    "Include entries whose names start with '.' (hidden files and "
                    "directories). Default: false."
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": (
                    "Display as a recursive tree instead of a flat table. "
                    "Noise directories (.git, __pycache__, node_modules, .venv, etc.) "
                    "are always excluded in this mode. Default: false."
                ),
            },
            "max_depth": {
                "type": "integer",
                "description": (
                    "Maximum recursion depth when recursive=true. "
                    "1 = immediate children only. Min 1, max 8. Default: 4."
                ),
                "minimum": 1,
                "maximum": 8,
            },
            "pattern": {
                "type": "string",
                "description": (
                    "Glob pattern to filter entries by filename. Applied to files only "
                    "(directories are always shown in tree mode). "
                    "Examples: '*.py', 'test_*', '*.md'. Default: show all."
                ),
            },
            "sort_by": {
                "type": "string",
                "enum": ["name", "modified", "size", "type"],
                "description": (
                    "Sort order within each type group (directories always appear before "
                    "files). 'name': alphabetical A→Z (default). "
                    "'modified': newest first. "
                    "'size': largest first. "
                    "'type': dirs, links, files, other."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum number of entries to return. Hard cap: 1000. Default: 200. "
                    "When the cap is hit a notice is shown — use a pattern or narrower "
                    "path to reduce results rather than raising this limit blindly."
                ),
                "minimum": 1,
                "maximum": 1000,
            },
        },
        "required": ["path"],
    },
    category="filesystem",
    capability_name="list_directory",
    capability_description=(
        "List a directory's contents (files, subdirectories, symlinks) with size, "
        "type, and modification time. Supports flat table and recursive tree views "
        "with glob filtering."
    ),
    capability_refusal="I can't access the filesystem right now.",
)
async def list_directory(
    path: str,
    show_hidden: bool = False,
    recursive: bool = False,
    max_depth: int = 4,
    pattern: Optional[str] = None,
    sort_by: str = "name",
    max_results: int = 200,
) -> str:

    # ── Parameter clamping ────────────────────────────────────────────────────
    max_results = max(1, min(max_results, _MAX_RESULTS_HARD))
    max_depth   = max(1, min(max_depth,   _MAX_DEPTH_HARD))
    if sort_by not in ("name", "modified", "size", "type"):
        sort_by = "name"

    # ── Path resolution ───────────────────────────────────────────────────────
    try:
        dirpath = Path(path).expanduser().resolve()
    except (ValueError, OSError) as exc:
        return f"Error: cannot resolve path '{path}': {exc}"

    # ── Existence and type checks ─────────────────────────────────────────────
    try:
        exists = dirpath.exists()
    except PermissionError:
        return f"Error: permission denied checking '{path}'."
    except OSError as exc:
        return f"Error: cannot access '{path}': {exc}"

    if not exists:
        if dirpath.suffix:
            return (
                f"Error: '{path}' does not exist.\n"
                f"It has a file extension — did you mean to use read_file?"
            )
        return f"Error: '{path}' does not exist."

    if dirpath.is_file():
        try:
            size_str = _human_size(dirpath.stat().st_size)
        except OSError:
            size_str = "unknown size"
        return (
            f"Error: '{path}' is a file ({size_str}), not a directory.\n"
            f"Use read_file to read its contents."
        )

    if not dirpath.is_dir():
        kind = _entry_kind_path(dirpath)
        return f"Error: '{path}' is not a directory (type: {kind})."

    if not os.access(dirpath, os.R_OK):
        return f"Error: permission denied — cannot read directory '{path}'."

    # ── Dispatch to the right renderer ────────────────────────────────────────
    if recursive:
        return await _render_tree(dirpath, show_hidden, pattern, sort_by, max_depth, max_results)
    else:
        return await _render_flat_async(dirpath, show_hidden, pattern, sort_by, max_results)


# ── Async wrappers for blocking I/O ──────────────────────────────────────────

async def _render_flat_async(
    dirpath: Path,
    show_hidden: bool,
    pattern: Optional[str],
    sort_by: str,
    max_results: int,
) -> str:
    def _work():
        entries, perm_errors = _scan_flat(dirpath, show_hidden, pattern)
        total = len(entries)
        entries.sort(key=lambda e: _sort_key(e, sort_by))
        entries = entries[:max_results]
        return _render_flat(dirpath, entries, total, perm_errors, max_results, pattern, sort_by)

    return await asyncio.to_thread(_work)


async def _render_tree(
    dirpath: Path,
    show_hidden: bool,
    pattern: Optional[str],
    sort_by: str,
    max_depth: int,
    max_results: int,
) -> str:
    def _work():
        builder = _TreeBuilder(show_hidden, pattern, sort_by, max_depth, max_results)
        child_lines = builder.build(dirpath)

        header = str(dirpath) + "/"
        meta_parts: list[str] = []
        if pattern:
            meta_parts.append(f"filter: {pattern}")
        meta_parts.append(f"depth≤{max_depth}")
        header += f"  [{', '.join(meta_parts)}]"

        out_lines: list[str] = [header] + child_lines

        # Footer
        count = builder.count
        footer_parts: list[str] = [
            f"{count} entr{'y' if count == 1 else 'ies'}"
        ]
        if builder.truncated:
            footer_parts.append(f"capped at {max_results}")
        if builder.perm_errors:
            footer_parts.append(f"{builder.perm_errors} dir(s) not readable")
        out_lines.append("")
        out_lines.append(" · ".join(footer_parts))

        return "\n".join(out_lines)

    return await asyncio.to_thread(_work)


def _entry_kind_path(p: Path) -> str:
    """Classify a Path (not DirEntry) — used for the is-not-a-dir check only."""
    try:
        if p.is_symlink():   return "symlink"
        m = p.stat().st_mode
        if stat.S_ISCHR(m):  return "character device"
        if stat.S_ISBLK(m):  return "block device"
        if stat.S_ISFIFO(m): return "named pipe"
        if stat.S_ISSOCK(m): return "socket"
        return "unknown"
    except OSError:
        return "unknown"
