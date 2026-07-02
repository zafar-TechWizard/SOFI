"""
BRAIN/tools/fs_find.py — find_files tool

Glob-pattern file discovery with:
  - Automatic pruning of noise directories (.git, node_modules, __pycache__, etc.)
  - Symlink-loop detection (safe follow_symlinks mode)
  - Depth limiting
  - Per-directory PermissionError resilience (skip & log, never crash)
  - Rich per-file metadata (absolute path, size, modified age)
  - Sorted newest-first by default so the most relevant results lead

Auto-discovered by _auto_register_tools via the module-level `register` alias.
"""

import asyncio
import fnmatch
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger("sofi.brain.tools.fs")


# ── Directory skip-lists ──────────────────────────────────────────────────────
#
# Applied during os.walk to prevent descending into irrelevant trees.
# Pruning here is far faster than descending and filtering after the fact
# (node_modules alone can contain 200k+ files).

_SKIP_EXACT: frozenset = frozenset({
    # Version control
    ".git", ".hg", ".svn", ".bzr",
    # Python bytecode / tooling cache
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".nox", ".ruff_cache",
    # Python virtual environments
    "venv", ".venv", "env",
    # JavaScript / Node.js
    "node_modules", ".next", ".nuxt", ".turbo",
    ".parcel-cache", ".cache",
    # Build output
    "dist", "build", "out", "target",
    # Language-specific vendor directories
    ".cargo", "vendor", ".gradle", ".m2",
    # Test coverage
    "coverage", ".nyc_output", "htmlcov",
    # Python packaging artefacts
    ".eggs",
    # SOFi internal (backup bin — never search inside it)
    "backup",
    # IDEs
    ".idea", ".vscode", ".fleet",
})

# Glob patterns matched against directory names (not full paths)
_SKIP_PATTERNS: tuple = (
    "*.egg-info",
    "*.dist-info",
    "__*__",          # __pycache__, __MACOSX__, etc.
)

# ── Hard limits ───────────────────────────────────────────────────────────────

_HARD_MAX: int = 500    # absolute ceiling — never send more than this to an LLM
_DEFAULT_MAX: int = 200


# ── Formatting helpers ────────────────────────────────────────────────────────

def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    n /= 1024
    if n < 1024:
        return f"{n:.1f} KB"
    n /= 1024
    if n < 1024:
        return f"{n:.1f} MB"
    n /= 1024
    return f"{n:.1f} GB"


def _human_age(ts: float) -> str:
    """Wall-clock age as a compact human string: '3h ago', '2d ago', etc."""
    d = time.time() - ts
    if d < 0:
        return "future"
    if d < 60:
        return "just now"
    if d < 3_600:
        m = int(d / 60)
        return f"{m}m ago"
    if d < 86_400:
        h = int(d / 3_600)
        return f"{h}h ago"
    if d < 7 * 86_400:
        days = int(d / 86_400)
        return f"{days}d ago"
    if d < 30 * 86_400:
        w = int(d / (7 * 86_400))
        return f"{w}w ago"
    if d < 365 * 86_400:
        mo = int(d / (30 * 86_400))
        return f"{mo}mo ago"
    y = int(d / (365 * 86_400))
    return f"{y}y ago"


# ── Directory pruning ─────────────────────────────────────────────────────────

def _keep_dir(name: str, extra_exclude: frozenset, include_hidden: bool) -> bool:
    """Return True if os.walk should descend into this directory."""
    if not include_hidden and name.startswith("."):
        return False
    if name in _SKIP_EXACT:
        return False
    if name in extra_exclude:
        return False
    for pat in _SKIP_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return False
    return True


# ── Pattern compilation ───────────────────────────────────────────────────────

def _glob_to_regex(pattern: str) -> re.Pattern:
    """
    Compile a glob pattern that may contain ``**`` to a regex.

    Glob → regex rules
    ------------------
    ``**``   → ``(?:.+/)?``   zero-or-more path components (with trailing /)
    ``*``    → ``[^/]*``      any chars except path separator
    ``?``    → ``[^/]``       any single char except path separator
    ``.``    → ``\\.``        literal dot
    others   → re.escape(c)

    Case sensitivity matches the platform default (case-insensitive on Windows).
    """
    buf: List[str] = []
    i = 0
    n = len(pattern)

    while i < n:
        c = pattern[i]
        if c == "*" and i + 1 < n and pattern[i + 1] == "*":
            # "**" — matches zero or more path components
            buf.append("(?:.+/)?")
            i += 2
            if i < n and pattern[i] == "/":
                i += 1   # consume the trailing slash ("**/")
        elif c == "*":
            buf.append("[^/]*")
            i += 1
        elif c == "?":
            buf.append("[^/]")
            i += 1
        else:
            buf.append(re.escape(c))
            i += 1

    flags = re.IGNORECASE if os.name == "nt" else 0
    return re.compile("^" + "".join(buf) + "$", flags)


def _make_matcher(pattern: str) -> Callable[[str, str], bool]:
    """
    Compile a glob pattern into match(rel_posix, basename) -> bool.

    Pattern forms supported
    -----------------------
    ``*.py``           pure basename glob — matches at any depth
    ``**/*.py``        same as above (canonical recursive form, normalised)
    ``src/**/*.py``    path-relative with recursive wildcard
    ``test_*.py``      basename prefix/suffix match
    ``config.json``    exact filename match

    Note: pathlib.Path.match() does **not** support ``**`` before Python 3.12,
    so all path patterns are matched via a custom regex compiler.

    Case sensitivity follows the platform default:
      Windows → case-insensitive (both fnmatch and the regex use IGNORECASE)
      POSIX   → case-sensitive
    """
    norm = pattern.replace("\\", "/").strip("/")

    if not norm:
        raise ValueError("pattern must not be empty after normalisation")

    # "**/<expr>" with no further separator → pure basename match at any depth.
    # This is the single most-common form; handle it fast via fnmatch.
    if norm.startswith("**/") and "/" not in norm[3:]:
        bp = norm[3:]
        def _basename_match(rel: str, bn: str, _bp: str = bp) -> bool:
            return fnmatch.fnmatch(bn, _bp)
        return _basename_match

    # No "/" anywhere → pure basename match.
    if "/" not in norm:
        def _basename_only(rel: str, bn: str, _norm: str = norm) -> bool:
            return fnmatch.fnmatch(bn, _norm)
        return _basename_only

    # Path pattern — compile to regex (supports ** on Python 3.9–3.11+).
    try:
        compiled = _glob_to_regex(norm)
    except re.error as exc:
        raise ValueError(f"pattern {pattern!r} produced invalid regex: {exc}") from exc

    def _path_match(rel: str, bn: str, _rx: re.Pattern = compiled) -> bool:
        return bool(_rx.match(rel))
    return _path_match


# ── Entry builder ─────────────────────────────────────────────────────────────

def _build_entry(
    root: Path,
    path: Path,
    entry_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Stat *path* and return a metadata dict, or None on any OS error.

    The ``_mtime`` field is an internal float used for sorting;
    it is stripped from results before returning to callers.
    """
    try:
        st = path.stat()
    except OSError as exc:
        _log.debug("find_files | stat failed | path=%s err=%s", path, exc)
        return None

    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()

    mtime = st.st_mtime
    size  = st.st_size if entry_type == "file" else 0

    return {
        "path":           str(path),
        "relative_path":  rel,
        "name":           path.name,
        "extension":      path.suffix.lstrip(".") if entry_type == "file" else "",
        "type":           entry_type,
        "size_bytes":     size,
        "size_human":     _human_size(size) if entry_type == "file" else "—",
        "modified_iso":   datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        "modified_human": _human_age(mtime),
        "_mtime":         mtime,   # internal — stripped before output
    }


# ── Output formatter ──────────────────────────────────────────────────────────

def _format_output(
    matches: List[Dict],
    total_found: int,
    root: Path,
    pattern: str,
    elapsed_ms: float,
) -> str:
    n = len(matches)
    truncated = total_found > n

    if total_found == 0:
        return (
            f'No results for "{pattern}" under {root}  '
            f"({elapsed_ms:.0f}ms)\n"
            "Tips: check the root path, broaden the pattern, or set include_hidden=true."
        )

    suffix = ""
    if truncated:
        suffix = f" — showing {n} most-recent of {total_found}"

    header = (
        f'{total_found} match{"es" if total_found != 1 else ""} '
        f'for "{pattern}" under {root}{suffix}  ({elapsed_ms:.0f}ms)\n'
    )

    # Align size and age columns for readability
    size_w = max(len(m["size_human"]) for m in matches) + 1
    age_w  = max(len(m["modified_human"]) for m in matches) + 1

    lines = [header]
    for m in matches:
        size = m["size_human"].rjust(size_w)
        age  = m["modified_human"].rjust(age_w)
        lines.append(f"  {m['path']}  {size}  {age}")

    if truncated:
        lines.append(
            f"\n{total_found - n} more result(s) not shown. "
            f"Increase max_results (up to {_HARD_MAX}) or narrow your pattern."
        )

    return "\n".join(lines)


# ── Walk error handler ────────────────────────────────────────────────────────

def _on_walk_error(exc: OSError) -> None:
    """
    os.walk onerror callback — called when scandir() fails on a subdirectory.
    We log at DEBUG (PermissionError on system dirs is routine) and continue.
    A counter is not maintained here; the walk simply skips the offending dir.
    """
    _log.debug("find_files | scandir error (skipped): %s", exc)


# ── Main handler ──────────────────────────────────────────────────────────────

async def find_files(
    pattern: str,
    root: str = ".",
    file_type: str = "file",
    max_results: int = _DEFAULT_MAX,
    max_depth: Optional[int] = None,
    exclude_dirs: Optional[List[str]] = None,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
    sort_by: str = "modified",
    sort_order: str = "desc",
) -> str:
    """
    Walk *root* and return files (or directories) whose names/paths match *pattern*.

    Returns a formatted string containing one absolute path per match line,
    with size and modified-age annotations. Full absolute paths are used so
    the LLM can pass them directly to read_file or other tools without
    needing to resolve anything.
    """
    t0 = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────
    pattern = (pattern or "").strip()
    if not pattern:
        return "Error: pattern must not be empty."

    if file_type not in ("file", "dir", "any"):
        return f"Error: file_type must be 'file', 'dir', or 'any'; got {file_type!r}."

    try:
        max_results = max(1, min(int(max_results), _HARD_MAX))
    except (ValueError, TypeError):
        max_results = _DEFAULT_MAX

    if sort_by not in ("modified", "name", "path", "size"):
        sort_by = "modified"
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"

    if max_depth is not None:
        try:
            max_depth = max(1, int(max_depth))
        except (ValueError, TypeError):
            max_depth = None

    # ── Resolve root ──────────────────────────────────────────────────────
    try:
        root_path = Path(root).expanduser().resolve()
    except (OSError, ValueError) as exc:
        return f"Error: cannot resolve root path {root!r}: {exc}"

    if not root_path.exists():
        return f"Error: root path does not exist: {root_path}"
    if not root_path.is_dir():
        return f"Error: root is not a directory: {root_path}"
    if not os.access(str(root_path), os.R_OK):
        return f"Error: root is not readable (permission denied): {root_path}"

    # ── Extra exclusions ──────────────────────────────────────────────────
    extra_exclude: frozenset = frozenset(exclude_dirs) if exclude_dirs else frozenset()

    # ── Compile pattern matcher ───────────────────────────────────────────
    try:
        match_fn = _make_matcher(pattern)
    except ValueError as exc:
        return f"Error: invalid pattern {pattern!r}: {exc}"

    # ── Walk (blocking I/O — run in a thread) ────────────────────────────
    def _collect() -> tuple:
        _matches: List[Dict] = []
        _total_found: int    = 0
        _visited_real: set   = set()

        try:
            for dirpath_str, dirnames, filenames in os.walk(
                str(root_path),
                topdown=True,
                onerror=_on_walk_error,
                followlinks=follow_symlinks,
            ):
                dirpath = Path(dirpath_str)

                # ── Current depth ─────────────────────────────────────────
                try:
                    depth = len(dirpath.relative_to(root_path).parts)
                except ValueError:
                    depth = 0

                # ── Symlink loop detection ────────────────────────────────
                if follow_symlinks:
                    try:
                        real = dirpath.resolve()
                        if real in _visited_real:
                            _log.debug(
                                "find_files | symlink loop | path=%s → %s",
                                dirpath, real,
                            )
                            dirnames[:] = []
                            continue
                        _visited_real.add(real)
                    except OSError:
                        pass

                # ── Prune dirnames in-place ───────────────────────────────
                if max_depth is not None and depth >= max_depth:
                    dirnames[:] = []
                else:
                    dirnames[:] = [
                        d for d in sorted(dirnames)
                        if _keep_dir(d, extra_exclude, include_hidden)
                    ]

                # ── Collect candidates at this level ──────────────────────
                if file_type == "file":
                    candidates: List[tuple] = [(n, "file") for n in filenames]
                elif file_type == "dir":
                    candidates = [(n, "dir") for n in dirnames]
                else:  # "any"
                    candidates = (
                        [(n, "file") for n in filenames] +
                        [(n, "dir")  for n in dirnames]
                    )

                for name, etype in candidates:
                    if not include_hidden and name.startswith("."):
                        continue

                    fp = dirpath / name

                    try:
                        rel_posix = fp.relative_to(root_path).as_posix()
                    except ValueError:
                        rel_posix = fp.as_posix()

                    if not match_fn(rel_posix, name):
                        continue

                    _total_found += 1

                    if len(_matches) < max_results:
                        entry = _build_entry(root_path, fp, etype)
                        if entry is not None:
                            _matches.append(entry)

        except Exception as exc:
            _log.error(
                "find_files | walk error | root=%s pattern=%r: %s",
                root_path, pattern, exc,
            )
            if not _matches:
                raise   # propagate — caller returns error string

        return _matches, _total_found

    try:
        matches, total_found = await asyncio.to_thread(_collect)
    except Exception as exc:
        return f"Error during search: {exc}"

    # ── Sort ──────────────────────────────────────────────────────────────
    reverse = sort_order == "desc"
    if sort_by == "modified":
        matches.sort(key=lambda e: e["_mtime"], reverse=reverse)
    elif sort_by == "name":
        matches.sort(key=lambda e: e["name"].lower(), reverse=reverse)
    elif sort_by == "path":
        matches.sort(key=lambda e: e["relative_path"].lower(), reverse=reverse)
    elif sort_by == "size":
        matches.sort(key=lambda e: e["size_bytes"], reverse=reverse)

    # Strip internal sort key before returning
    for m in matches:
        m.pop("_mtime", None)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    _log.debug(
        "find_files | done | pattern=%r root=%s total=%d returned=%d elapsed=%.0fms",
        pattern, root_path, total_found, len(matches), elapsed_ms,
    )

    return _format_output(
        matches=matches,
        total_found=total_found,
        root=root_path,
        pattern=pattern,
        elapsed_ms=elapsed_ms,
    )


# ── Registration ──────────────────────────────────────────────────────────────

def register(registry) -> None:
    from BRAIN.tools.registry import ToolEntry

    registry.register(ToolEntry(
        name="find_files",
        description=(
            "Locate files (or directories) matching a glob pattern under a directory tree.\n\n"
            "Returns one absolute path per line, sorted newest-modified first, with size "
            "and age annotations. Absolute paths can be passed directly to read_file or "
            "other tools without any path resolution.\n\n"
            "PATTERN FORMS:\n"
            "  *.py             any Python file at any depth\n"
            "  **/*.ts          any TypeScript file at any depth (explicit form)\n"
            "  src/**/*.tsx     React/TSX files only under src/\n"
            "  test_*.py        files whose names start with test_\n"
            "  config.json      exact filename anywhere in the tree\n"
            "  *.{py,js}        NOT supported — call twice with each extension\n\n"
            "AUTO-EXCLUDED (always skipped for speed):\n"
            "  node_modules, __pycache__, .git, venv, .venv, dist, build, "
            "target, .mypy_cache, .pytest_cache, .tox, coverage, and more.\n\n"
            "DEFAULTS:\n"
            "  root='.'  file_type='file'  max_results=200  sort newest-first\n"
            "  Hidden files/dirs (starting with '.') are excluded by default.\n\n"
            "WHEN TO USE:\n"
            "  • Before read_file when the exact path is unknown\n"
            "  • To discover all files of a type in a project\n"
            "  • Before search_in_files when looking for files by name not content"
        ),
        schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match filenames or paths. "
                        "Examples: '*.py', '**/*.ts', 'src/**/*.tsx', 'test_*.py', 'requirements.txt'"
                    ),
                },
                "root": {
                    "type": "string",
                    "description": (
                        "Directory to search from. Accepts '.', relative paths, "
                        "absolute paths, and '~' for home directory. Default: '.'"
                    ),
                    "default": ".",
                },
                "file_type": {
                    "type": "string",
                    "enum": ["file", "dir", "any"],
                    "description": (
                        "'file' matches regular files only (default). "
                        "'dir' matches directories. "
                        "'any' matches both."
                    ),
                    "default": "file",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of results to return. Range: 1–{_HARD_MAX}. "
                        f"Default: {_DEFAULT_MAX}. "
                        "If the total exceeds this, results are sorted first and the "
                        "most-relevant (per sort_by) are returned."
                    ),
                    "default": _DEFAULT_MAX,
                },
                "max_depth": {
                    "type": "integer",
                    "description": (
                        "Maximum directory depth to descend. "
                        "1 = root directory only, 2 = one level of subdirectories, etc. "
                        "Omit for unlimited depth (default)."
                    ),
                },
                "exclude_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Additional directory names to skip during traversal (exact match). "
                        "The built-in skip-list already covers common noise directories. "
                        "Use this for project-specific directories."
                    ),
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": (
                        "Include files and directories whose names start with '.'. "
                        "Default: false. Set true to search .github, .claude, .env files, etc."
                    ),
                    "default": False,
                },
                "follow_symlinks": {
                    "type": "boolean",
                    "description": (
                        "Follow symbolic links when walking the tree. "
                        "Loop detection is active — cycles are detected and skipped. "
                        "Default: false."
                    ),
                    "default": False,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["modified", "name", "path", "size"],
                    "description": (
                        "Sort results by this field. "
                        "'modified' (default) = most recently changed first. "
                        "'size' = largest first. "
                        "'name' / 'path' = alphabetical."
                    ),
                    "default": "modified",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": (
                        "Sort direction. 'desc' (default) = newest/largest/Z-first. "
                        "'asc' = oldest/smallest/A-first."
                    ),
                    "default": "desc",
                },
            },
            "required": ["pattern"],
        },
        handler=find_files,
        timeout=30.0,
        category="filesystem",
        capability_name="find_files",
        capability_description=(
            "Locate files and directories by glob pattern — "
            "use before read_file when the exact path is unknown."
        ),
        capability_refusal="I can't search the filesystem right now.",
    ))
