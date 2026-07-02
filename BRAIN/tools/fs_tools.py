"""
BRAIN/tools/fs_tools.py — Filesystem tools for SOFi

Read tools (always free, no workspace enforcement):
  search_files     Find files by name pattern (simple glob, filename only)
  search_in_files  Full-text / regex search across a directory tree
  file_info        Rich metadata for any path (file, dir, symlink, missing)

  Note: read_file is in read_file.py — a superior line-numbered, streaming
  implementation. This module intentionally does not register read_file.

Write tools (workspace-enforced — external paths go to active/ copy):
  write_file       Create or overwrite a file
  patch_file       Targeted string replacement inside an existing file
"""

import asyncio
import fnmatch
import json
import logging
import mimetypes
import os
import re
import shutil
import time
from asyncio import create_subprocess_exec, wait_for
from asyncio import TimeoutError as _ATimeoutError
from asyncio.subprocess import PIPE
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger("sofi.brain.tools.fs")

_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()   # assistant/
_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024                             # 5 MB — read cap

# ── Shared skip lists for search tools ───────────────────────────────────────

# Never recurse into these directory names.
_SKIP_DIRS: frozenset = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".npm", ".yarn", ".pnpm",
    "venv", ".venv", "env",
    "dist", "build", "out", "target", "output",
    ".tox", ".eggs", "htmlcov",
    ".idea", ".vscode", ".vs",
    "sofi-workspace",      # never search our own workspace/backup dirs
    ".temp",
})

# Skip files with these extensions — binary / compiled / media.
_SKIP_EXTENSIONS: frozenset = frozenset({
    ".pyc", ".pyo", ".pyd",
    ".so", ".dll", ".dylib", ".exe", ".bin", ".obj", ".o",
    ".class", ".jar", ".war",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".svg", ".ico", ".bmp",
    ".mp3", ".wav", ".ogg", ".flac", ".aac",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".whl", ".egg",
})

# Files larger than this are skipped in content-search (avoids loading huge files).
_MAX_FILE_SEARCH_BYTES: int = 10 * 1024 * 1024   # 10 MB

# Cached ripgrep availability (checked once).
_RG_AVAILABLE: Optional[bool] = None


def _rg_available() -> bool:
    global _RG_AVAILABLE
    if _RG_AVAILABLE is None:
        _RG_AVAILABLE = shutil.which("rg") is not None
        _log.debug("ripgrep available: %s", _RG_AVAILABLE)
    return _RG_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════
# search_files  (filename glob — finds files by name pattern)
# ═══════════════════════════════════════════════════════════════════════════

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

        def _search():
            found = []
            for root, dirs, files in os.walk(str(p)):
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

        matches, capped = await asyncio.to_thread(_search)

        if not matches:
            desc = f"pattern '{pattern}'"
            if content_search:
                desc += f" containing '{content_search}'"
            return f"No files found in {p} matching {desc}"

        lines = [f"Found {len(matches)} file(s) in {p}:"]
        for m in matches:
            lines.append(f"  {m}")
        if capped:
            lines.append(f"\n(Capped at {max_results} — narrow the pattern for precision)")

        return "\n".join(lines)

    except Exception as exc:
        _log.error("search_files error dir=%s: %s", directory, exc)
        return f"Error searching files: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# search_in_files  (full-text / regex search across file contents)
# ═══════════════════════════════════════════════════════════════════════════

async def search_in_files(
    root: str,
    pattern: str,
    file_glob: str = "*",
    case_sensitive: bool = False,
    context_lines: int = 1,
    max_results: int = 50,
    include_hidden: bool = False,
) -> str:
    """
    Search for a text or regex pattern inside files under *root*.

    Returns a human-readable report with file paths, line numbers, matching
    lines, and surrounding context. Designed for LLM consumption — readable
    without post-processing.

    Engine selection:
      • ripgrep (rg) when available — 10-50× faster, handles gitignore,
        encoding detection, and binary sniffing natively.
      • Pure Python fallback — functionally equivalent, uses os.walk with
        symlink-loop protection and null-byte binary detection.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    pattern = pattern.strip()
    if not pattern:
        return "Error: search pattern cannot be empty."

    root_path = Path(root).resolve()
    if not root_path.exists():
        return f"Error: path does not exist: {root_path}"

    try:
        regex_flags = 0 if case_sensitive else re.IGNORECASE
        compiled    = re.compile(pattern, regex_flags)
    except re.error as exc:
        return (
            f"Error: invalid regex — {exc}\n"
            "Tip: escape special chars with \\ "
            "or search a plain word ('def my_func', no special chars needed)."
        )

    context_lines = max(0, min(int(context_lines), 5))
    max_results   = max(1, min(int(max_results), 200))

    # ── Run search ────────────────────────────────────────────────────────
    try:
        if _rg_available():
            try:
                matches, files_searched, truncated = await _rg_search(
                    root_path, pattern, file_glob, case_sensitive,
                    context_lines, max_results, include_hidden,
                )
            except Exception as exc:
                _log.warning("rg search failed (%s) — falling back to Python", exc)
                matches, files_searched, truncated = await asyncio.to_thread(
                    _py_search,
                    root_path, compiled, file_glob,
                    context_lines, max_results, include_hidden,
                )
        else:
            matches, files_searched, truncated = await asyncio.to_thread(
                _py_search,
                root_path, compiled, file_glob,
                context_lines, max_results, include_hidden,
            )
    except Exception as exc:
        _log.exception("search_in_files: unexpected error")
        return f"Search failed: {exc}"

    return _fmt_search(
        root_path, pattern, file_glob, case_sensitive,
        matches, files_searched, truncated, max_results,
    )


# ── ripgrep backend ───────────────────────────────────────────────────────────

def _build_rg_cmd(
    pattern: str, file_glob: str, case_sensitive: bool,
    context_lines: int, include_hidden: bool, root: str,
) -> List[str]:
    cmd = ["rg", "--json", f"--context={context_lines}", "--no-messages"]

    if not case_sensitive:
        cmd.append("--ignore-case")
    if include_hidden:
        cmd.append("--hidden")
    if file_glob and file_glob != "*":
        cmd.extend(["--glob", file_glob])

    for d in sorted(_SKIP_DIRS):
        cmd.extend(["--glob", f"!**/{d}"])
        cmd.extend(["--glob", f"!**/{d}/**"])

    cmd.extend(["--", pattern, root])
    return cmd


async def _rg_search(
    root: Path, pattern: str, file_glob: str, case_sensitive: bool,
    context_lines: int, max_results: int, include_hidden: bool,
) -> Tuple[List[dict], int, bool]:
    """Run rg and parse ndjson output into match dicts."""
    cmd  = _build_rg_cmd(pattern, file_glob, case_sensitive, context_lines, include_hidden, str(root))
    proc = await create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)

    try:
        stdout, stderr = await wait_for(proc.communicate(), timeout=30.0)
    except _ATimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError("rg timed out after 30 s")

    if proc.returncode == 2:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"rg error: {err or '(no message)'}")

    matches: List[dict] = []
    files_searched      = 0
    truncated           = False

    # State machine: context events before a match → context_before;
    #                context events after a match   → context_after.
    ctx_before: List[Tuple[int, str]] = []
    last_match: Optional[dict]        = None
    after_left: int                   = 0

    for raw in stdout.decode("utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        ev_type: str = ev.get("type", "")
        data: dict   = ev.get("data", {})

        if ev_type == "begin":
            ctx_before = []
            last_match = None
            after_left = 0

        elif ev_type == "context":
            ln   = data.get("line_number", 0)
            text = data.get("lines", {}).get("text", "").rstrip("\r\n")
            if after_left > 0 and last_match is not None:
                last_match["context_after"].append((ln, text))
                after_left -= 1
                if after_left == 0:
                    last_match = None
            else:
                ctx_before.append((ln, text))
                if len(ctx_before) > context_lines:
                    ctx_before = ctx_before[-context_lines:]

        elif ev_type == "match":
            if len(matches) >= max_results:
                truncated = True
                continue   # keep reading for files_searched count in summary
            ln   = data.get("line_number", 0)
            text = data.get("lines", {}).get("text", "").rstrip("\r\n")
            m: dict = {
                "file":           Path(data.get("path", {}).get("text", "")),
                "line_number":    ln,
                "line_content":   text,
                "context_before": list(ctx_before),
                "context_after":  [],
            }
            matches.append(m)
            last_match = m
            ctx_before = []
            after_left = context_lines

        elif ev_type == "summary":
            files_searched = data.get("stats", {}).get("searches", 0)

    return matches, files_searched, truncated


# ── pure-Python backend ───────────────────────────────────────────────────────

def _py_search(
    root: Path, compiled: re.Pattern, file_glob: str,
    context_lines: int, max_results: int, include_hidden: bool,
) -> Tuple[List[dict], int, bool]:
    matches: List[dict] = []
    files_searched      = 0
    truncated           = False

    for filepath in _iter_searchable(root, file_glob, include_hidden):
        if truncated:
            break
        files_searched += 1
        try:
            for m in _search_one_file(filepath, compiled, context_lines):
                if len(matches) >= max_results:
                    truncated = True
                    break
                matches.append(m)
        except Exception as exc:
            _log.debug("search_in_files: skip %s — %s", filepath, exc)

    return matches, files_searched, truncated


def _iter_searchable(root: Path, file_glob: str, include_hidden: bool):
    """
    Yield text-file candidates under *root* using os.walk(followlinks=False).

    followlinks=False prevents symlink loops on circular directory structures.
    Skip dirs are pruned in-place so os.walk never descends into them.
    """
    if root.is_file():
        if root.suffix.lower() not in _SKIP_EXTENSIONS:
            yield root
        return

    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=False):
        # Prune — modifies the list os.walk uses for recursion.
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and (include_hidden or not d.startswith("."))
        ]

        for fname in filenames:
            if not include_hidden and fname.startswith("."):
                continue
            _, ext = os.path.splitext(fname)
            if ext.lower() in _SKIP_EXTENSIONS:
                continue
            if file_glob and file_glob != "*":
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
            yield Path(dirpath) / fname


def _search_one_file(
    filepath: Path, compiled: re.Pattern, context_lines: int,
) -> List[dict]:
    """Search a single file. Returns match dicts. Raises on I/O errors."""
    size = filepath.stat().st_size
    if size == 0 or size > _MAX_FILE_SEARCH_BYTES:
        return []

    # Binary sniff: null bytes in first 8 KB → binary file, skip.
    chunk = filepath.read_bytes()[:8192]
    if b"\x00" in chunk:
        return []

    lines   = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    results = []

    for i, line in enumerate(lines):
        if not compiled.search(line):
            continue

        before_start = max(0, i - context_lines)
        after_end    = min(len(lines), i + context_lines + 1)

        results.append({
            "file":           filepath,
            "line_number":    i + 1,
            "line_content":   line,
            "context_before": [
                (before_start + j + 1, lines[before_start + j])
                for j in range(i - before_start)
            ],
            "context_after":  [
                (i + j + 2, lines[i + j + 1])
                for j in range(after_end - i - 1)
            ],
        })

    return results


# ── search_in_files formatting ────────────────────────────────────────────────

def _fmt_search(
    root: Path, pattern: str, file_glob: str, case_sensitive: bool,
    matches: List[dict], files_searched: int, truncated: bool, max_results: int,
) -> str:
    if not matches:
        glob_note   = f" in {file_glob!r}" if file_glob not in ("*", "") else ""
        case_note   = " (case-sensitive)" if case_sensitive else ""
        search_note = f"{files_searched} file(s) searched" if files_searched else "0 files searched"
        return (
            f'No matches for "{pattern}"{case_note}{glob_note}.\n'
            f'{search_note} under {root}'
        )

    # Group by file, preserve first-match ordering.
    files_order: List[Path] = []
    by_file: Dict[Path, List[dict]] = {}
    for m in matches:
        fp = m["file"]
        if fp not in by_file:
            by_file[fp] = []
            files_order.append(fp)
        by_file[fp].append(m)

    case_note   = " (case-sensitive)" if case_sensitive else ""
    glob_note   = f" | {file_glob!r}" if file_glob not in ("*", "") else ""
    trunc_note  = f"  [first {max_results} shown — more exist]" if truncated else ""
    search_note = f" | {files_searched} file(s) searched" if files_searched else ""

    out: List[str] = [
        f'"{pattern}"{case_note}{glob_note} — '
        f'{len(matches)} match(es) in {len(files_order)} file(s)'
        f'{trunc_note}{search_note}',
        "",
    ]

    for file_path in files_order:
        file_matches = by_file[file_path]
        try:
            display = str(file_path.relative_to(root))
        except ValueError:
            display = str(file_path)

        out.append(f"── {display} ──")

        last_shown = -1
        for m in file_matches:
            first_ctx = (
                m["context_before"][0][0] if m["context_before"] else m["line_number"]
            )
            # Gap divider between non-adjacent match blocks in the same file.
            if last_shown >= 0 and first_ctx > last_shown + 1:
                out.append("     ···")

            for ctx_ln, ctx_text in m["context_before"]:
                out.append(f"  {ctx_ln:>5} │ {ctx_text}")

            out.append(f"► {m['line_number']:>5} │ {m['line_content']}")

            for ctx_ln, ctx_text in m["context_after"]:
                out.append(f"  {ctx_ln:>5} │ {ctx_text}")

            last_shown = (
                m["context_after"][-1][0] if m["context_after"] else m["line_number"]
            )

        out.append("")

    if truncated:
        out.append(
            f"Results capped at {max_results}. "
            "Use a more specific pattern or increase max_results (up to 200)."
        )

    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════════
# file_info
# ═══════════════════════════════════════════════════════════════════════════

async def file_info(path: str) -> str:
    """
    Return rich metadata about *path*.

    Handles: files, directories, symlinks (intact or broken), missing paths,
    permission-denied cases, and special filesystem objects. Never raises.
    """
    try:
        return await asyncio.to_thread(_file_info_impl, path)
    except Exception as exc:
        _log.exception("file_info: unexpected error for %r", path)
        return f"Error inspecting {path!r}: {exc}"


def _file_info_impl(path: str) -> str:
    p   = Path(path).resolve()
    now = time.time()

    is_symlink = p.is_symlink()

    # ── Does it exist? ────────────────────────────────────────────────────
    if not p.exists() and not is_symlink:
        parent_ok = p.parent.exists()
        lines = [f"path:     {p}", "exists:   no"]
        lines.append(
            f"parent:   {'exists' if parent_ok else 'also missing'}  ({p.parent})"
        )
        return "\n".join(lines)

    # ── Read symlink target ───────────────────────────────────────────────
    symlink_target: Optional[str] = None
    if is_symlink:
        try:
            symlink_target = os.readlink(str(p))
        except OSError:
            symlink_target = "(unreadable)"

    # ── stat ─────────────────────────────────────────────────────────────
    try:
        st = p.stat() if p.exists() else p.lstat()
    except PermissionError:
        prefix = "yes (broken symlink)" if (is_symlink and not p.exists()) else "yes"
        return f"path:     {p}\nexists:   {prefix}\nstat:     permission denied"
    except OSError as exc:
        return f"path:     {p}\nexists:   unknown\nstat:     {exc}"

    if p.is_dir():
        return _fmt_dir(p, st, is_symlink, symlink_target, now)
    return _fmt_file(p, st, is_symlink, symlink_target, now)


def _fmt_file(
    p: Path, st, is_symlink: bool, symlink_target: Optional[str], now: float,
) -> str:
    size                  = st.st_size
    is_text, enc_hint     = _detect_text(p, size)
    mime_type, _          = mimetypes.guess_type(str(p))
    mime_type             = mime_type or ("text/plain" if is_text else "application/octet-stream")

    line_count: Optional[int] = None
    if is_text and 0 < size <= _MAX_FILE_SEARCH_BYTES:
        try:
            content    = p.read_text(encoding="utf-8", errors="replace")
            line_count = content.count("\n") + (
                1 if content and not content.endswith("\n") else 0
            )
        except (PermissionError, OSError):
            pass

    readable  = os.access(str(p), os.R_OK)
    writable  = os.access(str(p), os.W_OK)

    out: List[str] = []

    if is_symlink and symlink_target:
        out.append(f"path:      {p}  →  {symlink_target}")
        out.append(f"type:      symlink → {'text file' if is_text else 'binary file'}")
    else:
        out.append(f"path:      {p}")
        out.append(f"type:      {'text file' if is_text else 'binary file'}")

    out.append(f"size:      {size:,} bytes  ({_human_size(size)})")

    if line_count is not None:
        out.append(f"lines:     {line_count:,}")
    elif not is_text:
        out.append("lines:     —  (binary)")
    else:
        out.append(f"lines:     —  (file > {_human_size(_MAX_FILE_SEARCH_BYTES)}, not counted)")

    out.append(f"mime:      {mime_type}")

    if enc_hint:
        out.append(f"encoding:  {enc_hint}")

    mtime_str = _fmt_ts(st.st_mtime)
    mtime_ago = _time_ago(now - st.st_mtime)
    out.append(f"modified:  {mtime_str}  ({mtime_ago})")

    if hasattr(st, "st_atime") and abs(st.st_atime - st.st_mtime) > 120:
        out.append(f"accessed:  {_fmt_ts(st.st_atime)}")

    out.append(f"readable:  {'yes' if readable else 'no'}")
    out.append(f"writable:  {'yes' if writable else 'no'}")

    return "\n".join(out)


def _fmt_dir(
    p: Path, st, is_symlink: bool, symlink_target: Optional[str], now: float,
) -> str:
    n_files = n_dirs = n_other = 0
    child_err: Optional[str] = None

    try:
        for child in p.iterdir():
            if child.is_file(follow_symlinks=False):
                n_files += 1
            elif child.is_dir(follow_symlinks=False):
                n_dirs += 1
            else:
                n_other += 1
    except PermissionError:
        child_err = "permission denied"
    except OSError as exc:
        child_err = str(exc)

    readable = os.access(str(p), os.R_OK)
    writable  = os.access(str(p), os.W_OK)

    out: List[str] = []

    if is_symlink and symlink_target:
        out.append(f"path:      {p}/  →  {symlink_target}")
        out.append("type:      symlink → directory")
    else:
        out.append(f"path:      {p}/")
        out.append("type:      directory")

    if child_err:
        out.append(f"children:  ({child_err})")
    else:
        parts = []
        if n_files: parts.append(f"{n_files} file{'s' if n_files != 1 else ''}")
        if n_dirs:  parts.append(f"{n_dirs} director{'ies' if n_dirs != 1 else 'y'}")
        if n_other: parts.append(f"{n_other} other")
        out.append("children:  " + (", ".join(parts) if parts else "empty") + "  (direct)")

    mtime_str = _fmt_ts(st.st_mtime)
    mtime_ago = _time_ago(now - st.st_mtime)
    out.append(f"modified:  {mtime_str}  ({mtime_ago})")
    out.append(f"readable:  {'yes' if readable else 'no'}")
    out.append(f"writable:  {'yes' if writable else 'no'}")

    return "\n".join(out)


# ── file_info helpers ─────────────────────────────────────────────────────────

def _detect_text(p: Path, size: int) -> Tuple[bool, Optional[str]]:
    """
    Return (is_text, encoding_hint).

    Algorithm:
    1. Empty file → text/UTF-8.
    2. Read up to 8 KB; null byte → binary.
    3. Try strict UTF-8 decode → text/UTF-8.
    4. Latin-1 printable-ratio > 70 % → text/Latin-1.
    5. Else binary.
    """
    if size == 0:
        return True, "UTF-8"
    try:
        chunk = p.read_bytes()[:8192]
    except (PermissionError, OSError):
        return False, None

    if b"\x00" in chunk:
        return False, None

    try:
        chunk.decode("utf-8")
        return True, "UTF-8"
    except UnicodeDecodeError:
        pass

    try:
        printable = sum(1 for b in chunk if (32 <= b < 127) or b in (9, 10, 13))
        if printable / len(chunk) > 0.70:
            return True, "Latin-1 / Windows-1252"
    except Exception:
        pass

    return False, None


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    f = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        f /= 1024.0
        if f < 1024.0 or unit == "TB":
            return f"{f:.1f} {unit}" if f < 10 else f"{f:.0f} {unit}"
    return f"{f:.0f} TB"


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _time_ago(seconds: float) -> str:
    s = int(seconds)
    if s < 5:      return "just now"
    if s < 60:     return f"{s}s ago"
    if s < 3600:   return f"{s // 60}m ago"
    if s < 86400:  return f"{s // 3600}h ago"
    if s < 86400 * 30:  return f"{s // 86400}d ago"
    if s < 86400 * 365: return f"{s // (86400 * 30)}mo ago"
    return f"{s // (86400 * 365)}y ago"


# ═══════════════════════════════════════════════════════════════════════════
# Write utilities — shared by write_file and patch_file
# ═══════════════════════════════════════════════════════════════════════════

# OS system directories that are always blocked regardless of workspace policy.
_BLOCKED_WRITE_PREFIXES: tuple = (
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/boot", "/proc", "/sys", "/dev",
    "/System", "/Library", "/Applications",
)


def _check_write_blocked(p: Path) -> Tuple[bool, str]:
    """
    Return (blocked, reason).

    Blocks OS system directories unconditionally.
    Also blocks direct writes inside backup/ — that directory is managed
    exclusively by BackupManager; direct writes would corrupt the index.
    """
    s = str(p).replace("\\", "/")
    for prefix in _BLOCKED_WRITE_PREFIXES:
        if s.lower().startswith(prefix.lower().replace("\\", "/")):
            return True, f"Writing to system path is blocked: {p}"
    try:
        from BRAIN.tools.workspace import get_manager as _gwm
        bp = _gwm().backup_root.resolve()
        rp = p.resolve()
        if rp == bp or bp in rp.parents:
            return True, (
                "Direct writes to the backup directory are not allowed. "
                "Use delete_file / restore_backup instead."
            )
    except Exception:
        pass
    return False, ""


def _get_workspace_manager():
    """Lazy import — avoids circular imports at module-load time."""
    try:
        from BRAIN.tools.workspace import get_manager
        return get_manager()
    except Exception:
        return None


def _resolve_write_path(path: str) -> Tuple[Path, Path, bool]:
    """
    Apply workspace routing policy to *path*.

    Returns (original, write_target, is_workspace_copy).

    Internal path (inside sofi-workspace/) → write directly; write_target == original.
    External path → route to active/ mirror; original is never touched.
    """
    original = Path(path).expanduser().resolve()
    ws = _get_workspace_manager()
    if ws is None or ws.is_internal(original):
        return original, original, False
    return original, ws.to_active_copy(original), True


def _atomic_write_text(target: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Write *content* to *target* atomically via a sibling .sofi_tmp + os.replace().

    Guarantees:
      • Readers always see the old or new complete content — never a partial write.
      • Power failure between the write and the rename leaves the original intact.
      • On Windows, os.replace() is atomic at the NTFS level.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".sofi_tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(str(tmp), str(target))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Atomic byte-level write — used by patch_file to preserve the original encoding."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".sofi_tmp")
    try:
        tmp.write_bytes(data)
        os.replace(str(tmp), str(target))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _detect_encoding(raw: bytes) -> str:
    """
    Detect file encoding from raw bytes.

    Priority:
      1. UTF-8 BOM  → 'utf-8-sig'
      2. UTF-16 BOM → 'utf-16'
      3. Valid UTF-8 (no errors) → 'utf-8'
      4. Fallback → 'latin-1'  (never raises; any byte sequence is valid)
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _read_preserving(path: Path) -> Tuple[str, str, str]:
    """
    Read a file, detecting its encoding and dominant line-ending style.

    Returns (content, encoding, line_ending) where:
      content      — decoded text with \\n-normalised line endings
      encoding     — encoding to use when writing back
      line_ending  — original dominant style ('\\r\\n' or '\\n')

    Both encoding and line_ending must be passed back to _write_preserving
    so the file is restored byte-for-byte in terms of format.
    """
    raw      = path.read_bytes()
    encoding = _detect_encoding(raw)
    try:
        content = raw.decode(encoding)
    except Exception:
        content  = raw.decode("utf-8", errors="replace")
        encoding = "utf-8"

    # Count raw occurrences before any normalisation.
    crlf        = content.count("\r\n")
    lf          = content.count("\n") - crlf        # pure \n (not part of \r\n)
    line_ending = "\r\n" if crlf > lf else "\n"

    content = content.replace("\r\n", "\n")          # normalise internally
    return content, encoding, line_ending


def _write_preserving(target: Path, content: str, encoding: str, line_ending: str) -> None:
    """
    Write *content* (\\n-normalised) back to *target* using the original
    encoding and line-ending style detected by _read_preserving.
    """
    if line_ending == "\r\n":
        content = content.replace("\n", "\r\n")
    try:
        raw = content.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        raw = content.encode("utf-8", errors="replace")
    _atomic_write_bytes(target, raw)


def _line_number_at(content: str, char_idx: int) -> int:
    """Return the 1-based line number for a character index within *content*."""
    return content[:char_idx].count("\n") + 1


def _brief_line(text: str, max_chars: int = 80) -> str:
    """First non-blank line of *text*, truncated to *max_chars* characters."""
    line = next((l for l in text.split("\n") if l.strip()), text.split("\n")[0])
    return line if len(line) <= max_chars else line[:max_chars - 3] + "..."


def _near_miss_message(content: str, old_string: str) -> Optional[str]:
    """
    Check whether old_string matches after stripping trailing whitespace from
    each line.  This is the most common cause of patch failures — editors
    silently add or remove trailing spaces.

    Returns a diagnostic string with the *exact* text from the file at the
    match location (so the caller can copy-paste the correct old_string),
    or None when no near-match is found.
    """
    def _strip_trailing(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.split("\n"))

    norm_content = _strip_trailing(content)
    norm_old     = _strip_trailing(old_string)
    if not norm_old:
        return None

    count = norm_content.count(norm_old)
    if count == 0:
        return None

    idx      = norm_content.find(norm_old)
    line_num = norm_content[:idx].count("\n") + 1
    n_lines  = old_string.count("\n") + 1

    # Extract the actual (un-normalised) lines from the file so the
    # caller can see the exact whitespace that differs.
    file_lines   = content.split("\n")
    actual_lines = file_lines[line_num - 1 : line_num - 1 + n_lines]
    actual_text  = "\n".join(actual_lines)

    if count == 1:
        return (
            f"Near-match at line {line_num} — trailing whitespace differs.\n"
            f"Exact text in file:\n{'─' * 40}\n{actual_text}\n{'─' * 40}\n"
            f"Copy this text as old_string and retry."
        )
    return (
        f"{count} near-matches (trailing whitespace differs). "
        f"Increase specificity or check line endings."
    )


# ═══════════════════════════════════════════════════════════════════════════
# write_file
# ═══════════════════════════════════════════════════════════════════════════

async def write_file(
    path: str,
    content: str,
    mode: str = "overwrite",
    encoding: str = "utf-8",
) -> str:
    """
    Create or modify a file, respecting workspace routing policy.

    Workspace routing:
      • Internal path (inside sofi-workspace/) → write directly.
      • External path → write to active/ working copy; original is NEVER modified.
        Apply the working copy manually if you want to update the original.

    mode:
      overwrite  (default) — replace entire file; creates if it does not exist.
      create     — write only if the file does not exist (error if it does).
      append     — add content to the end of an existing file.

    All overwrites use an atomic tmp+replace strategy — power failure cannot
    leave a partially written file.  Parent directories are created automatically.
    """
    if mode not in ("create", "overwrite", "append"):
        return f"Error: invalid mode '{mode}'. Use: create, overwrite, or append."
    try:
        return await asyncio.to_thread(_write_file_impl, path, content, mode, encoding)
    except Exception as exc:
        _log.exception("write_file | path=%s", path)
        return f"Error writing file: {exc}"


def _write_file_impl(path: str, content: str, mode: str, encoding: str) -> str:
    original, target, is_copy = _resolve_write_path(path)

    blocked, reason = _check_write_blocked(original)
    if blocked:
        return f"Blocked: {reason}"

    # create mode: fail if target already exists.
    if mode == "create" and target.exists():
        label = (
            f"Working copy already exists: {target}"
            if is_copy else f"File already exists: {original}"
        )
        return f"{label}\nUse mode='overwrite' to replace it."

    if mode == "append":
        # Seed the working copy from the original before appending.
        if is_copy and not target.exists() and original.exists():
            try:
                import shutil as _sh
                target.parent.mkdir(parents=True, exist_ok=True)
                _sh.copy2(str(original), str(target))
            except Exception as exc:
                return f"Could not create working copy to append to: {exc}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding=encoding) as fh:
            fh.write(content)
        size = target.stat().st_size
        action_line = (
            f"Appended {len(content):,} chars to {target}  "
            f"(file now {size:,} bytes)"
        )
    else:
        # overwrite (and create on first write)
        existed = target.exists()
        _atomic_write_text(target, content, encoding)
        action       = "Updated" if existed else "Created"
        bytes_written = len(content.encode(encoding, errors="replace"))
        action_line  = (
            f"{action}: {target}  "
            f"({len(content):,} chars, {bytes_written:,} bytes)"
        )

    lines = [action_line]
    if is_copy:
        lines.append(
            f"↳ External path — working copy only (original untouched): {original}"
        )

    _log.info("write_file | path=%s mode=%s is_copy=%s", original, mode, is_copy)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# patch_file
# ═══════════════════════════════════════════════════════════════════════════

async def patch_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """
    Make a precise in-place edit: replace old_string with new_string.

    Workspace routing: same as write_file — external files get a working copy
    in active/; the original is never modified directly.

    Guarantees:
      • Fails clearly if old_string is not found — no silent no-ops.
      • Fails if old_string matches more than once and replace_all=False,
        reporting the line numbers of every occurrence.
      • Near-miss detection: if the mismatch is only trailing whitespace,
        shows the exact file text so the caller can self-correct without
        having to re-read the file.
      • Preserves original encoding (UTF-8 / Latin-1) and line endings
        (LF / CRLF) — byte-for-byte identical format on write-back.
      • Atomic write — power failure cannot corrupt the file.
    """
    if not old_string:
        return "Error: old_string cannot be empty."
    try:
        return await asyncio.to_thread(
            _patch_file_impl, path, old_string, new_string, replace_all,
        )
    except Exception as exc:
        _log.exception("patch_file | path=%s", path)
        return f"Error patching file: {exc}"


def _patch_file_impl(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> str:
    original, target, is_copy = _resolve_write_path(path)

    blocked, reason = _check_write_blocked(original)
    if blocked:
        return f"Blocked: {reason}"

    # Read from working copy if it exists, otherwise fall back to original.
    read_from = target if target.exists() else original
    if not read_from.exists():
        return f"File not found: {original}"
    if not read_from.is_file():
        return f"Not a file: {original}"

    # Seed working copy from original when the copy doesn't exist yet.
    if is_copy and not target.exists() and original.exists():
        try:
            import shutil as _sh
            target.parent.mkdir(parents=True, exist_ok=True)
            _sh.copy2(str(original), str(target))
            read_from = target
        except Exception as exc:
            return f"Could not create working copy: {exc}"

    # Read the file preserving its encoding and line-ending style.
    content, encoding, line_ending = _read_preserving(read_from)

    # Normalise old/new to \n so they match content (also \n-normalised).
    old_norm = old_string.replace("\r\n", "\n")
    new_norm = new_string.replace("\r\n", "\n")

    # ── Exact-match check ─────────────────────────────────────────────────
    count = content.count(old_norm)

    if count == 0:
        hint = _near_miss_message(content, old_norm)
        if hint:
            return f"old_string not found in {original}.\n{hint}"
        preview = content[:300].rstrip()
        return (
            f"old_string not found in {original}.\n"
            f"File starts with:\n{'─' * 40}\n{preview}\n{'─' * 40}\n"
            f"Tip: use read_file to see exact content, then retry with the copied text."
        )

    if count > 1 and not replace_all:
        # Report every line number so the caller can make old_string more specific.
        occurrences: List[int] = []
        pos = 0
        while True:
            idx = content.find(old_norm, pos)
            if idx == -1:
                break
            occurrences.append(_line_number_at(content, idx))
            pos = idx + 1
        lines_str = ", ".join(str(n) for n in occurrences)
        return (
            f"old_string appears {count} times in {original} "
            f"(at lines: {lines_str}).\n"
            f"Add more surrounding context to make it unique, "
            f"or pass replace_all=true to replace all {count} occurrences."
        )

    # ── Apply replacement ─────────────────────────────────────────────────
    replacements = count if replace_all else 1
    new_content  = content.replace(old_norm, new_norm, replacements)

    # Capture first match location for the success report.
    first_idx  = content.find(old_norm)
    first_line = _line_number_at(content, first_idx)

    # Write back, restoring the original encoding and line endings.
    _write_preserving(target, new_content, encoding, line_ending)

    # ── Build success report ──────────────────────────────────────────────
    removed_lines = old_norm.count("\n") + 1
    added_lines   = new_norm.count("\n") + 1
    char_delta    = (len(new_norm) - len(old_norm)) * replacements
    sign          = "+" if char_delta >= 0 else ""

    report_lines = [
        f"Patched: {target if is_copy else original}",
        f"  at line {first_line}"
        + (f"  ×{replacements} occurrences" if replacements > 1 else ""),
        f"  was: {_brief_line(old_norm)!r}",
        f"  now: {_brief_line(new_norm)!r}",
        f"  change: {removed_lines}→{added_lines} line(s)  ({sign}{char_delta} chars)",
    ]
    if is_copy:
        report_lines.append(
            f"↳ Working copy only — original untouched: {original}"
        )

    _log.info(
        "patch_file | path=%s line=%d replacements=%d is_copy=%s",
        original, first_line, replacements, is_copy,
    )
    return "\n".join(report_lines)


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

def register_fs_tools(registry) -> None:
    from BRAIN.tools.registry import ToolEntry

    # ── search_files ───────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="search_files",
        description=(
            "Find files by name pattern in a directory tree. "
            "pattern uses glob syntax: '*.py', '*.md', 'config*'. "
            "Optionally filter to files containing a specific string. "
            "Returns file paths. Use search_in_files to search inside file content."
        ),
        schema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Root directory to search (default: assistant workspace root)",
                    "default": ".",
                },
                "pattern": {
                    "type": "string",
                    "description": "Filename glob pattern e.g. '*.py', '*.json', 'brain*'",
                    "default": "*",
                },
                "content_search": {
                    "type": "string",
                    "description": "Optional: only return files whose content contains this text",
                    "default": "",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum file paths to return (default 20)",
                    "default": 20,
                },
            },
            "required": [],
        },
        handler=search_files,
        category="filesystem",
        capability_name="search_files",
        capability_description="Find files by name or content on the local filesystem.",
        capability_refusal="I can't search files right now.",
    ))

    # ── search_in_files ────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="search_in_files",
        description=(
            "Search for a text or regex pattern inside files under a directory. "
            "Returns matches with file path, line number, and surrounding context lines.\n\n"
            "Uses ripgrep (rg) when available — fast, gitignore-aware, binary-safe. "
            "Falls back to pure Python when rg is absent.\n\n"
            "Use for:\n"
            "• Finding where a function / class / variable is defined or used\n"
            "• Locating config values, env vars, error messages\n"
            "• Discovering all TODO / FIXME / HACK comments\n"
            "• Tracing a concept through an unfamiliar codebase\n\n"
            "Pattern is a regex. Plain words work as-is. "
            "Escape special chars (. * + ?) with \\ for literal search.\n\n"
            "Automatically skips: binary files, node_modules, __pycache__, .git, "
            "venv, dist, build, sofi-workspace, and other noise."
        ),
        schema={
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": (
                        "Absolute path to the directory (or single file) to search. "
                        "Tip: use the project root for broad searches, a subdirectory to narrow scope."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Search pattern (regex supported). Examples:\n"
                        "  'def my_func'          — literal function definition\n"
                        "  'TODO.*urgent'         — TODO with 'urgent' anywhere after\n"
                        "  'class\\s+\\w+Manager'  — any class ending in 'Manager'\n"
                        "  '\\bapi_key\\b'         — exact word 'api_key'"
                    ),
                },
                "file_glob": {
                    "type": "string",
                    "description": "Filename filter (glob). E.g. '*.py', '*.ts', '*.md'. Default '*' = all text files.",
                    "default": "*",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive match. Default false.",
                    "default": False,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context before and after each match (0–5, default 1).",
                    "default": 1,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return (1–200, default 50).",
                    "default": 50,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files/dirs (names starting with '.'). Default false.",
                    "default": False,
                },
            },
            "required": ["root", "pattern"],
        },
        handler=search_in_files,
        category="filesystem",
        capability_name="search_in_files",
        capability_description=(
            "Search for text or regex patterns across file contents — find definitions, "
            "usages, config values, and any concept across a codebase."
        ),
        capability_refusal="I can't search file contents right now.",
    ))

    # ── file_info ──────────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="file_info",
        description=(
            "Get rich metadata about any path without reading its content: "
            "type (file/dir/symlink), size, line count, MIME type, encoding, "
            "modification time, and read/write permissions.\n\n"
            "Works on any path — file, directory, symlink (intact or broken), "
            "or a path that doesn't exist yet. Never raises.\n\n"
            "Use before read_file to:\n"
            "• Confirm a file exists and is readable before opening it\n"
            "• Know the size and line count so you can plan how many chunks to read\n"
            "• Check write permission before creating or modifying a file\n"
            "• Distinguish text from binary (binary files can't be read_file'd)\n"
            "• Check modification time to gauge recency"
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to inspect (file, directory, symlink, or non-existent path).",
                },
            },
            "required": ["path"],
        },
        handler=file_info,
        category="filesystem",
        capability_name="file_info",
        capability_description=(
            "Get metadata about any path — size, type, line count, encoding, "
            "timestamps, permissions — without reading the file's content."
        ),
        capability_refusal="I can't inspect files right now.",
    ))

    # ── write_file ─────────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="write_file",
        description=(
            "Create or modify a file on the local filesystem.\n\n"
            "WORKSPACE ROUTING:\n"
            "  • Paths inside sofi-workspace/ → written directly.\n"
            "  • External paths → written to a working copy in active/; "
            "the original file is NEVER modified. Apply the copy manually if needed.\n\n"
            "MODES:\n"
            "  overwrite (default) — replace entire file; creates if it doesn't exist.\n"
            "  create  — write only if file doesn't exist (error if it does).\n"
            "  append  — add content to the end of an existing file.\n\n"
            "All writes are atomic (tmp+replace) — power failure cannot corrupt the file.\n"
            "Parent directories are created automatically."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the target file. "
                        "External paths are safely routed to a working copy."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["create", "overwrite", "append"],
                    "description": "Write mode. Default: overwrite.",
                    "default": "overwrite",
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding. Default: utf-8. Use latin-1 for legacy files.",
                    "default": "utf-8",
                },
            },
            "required": ["path", "content"],
        },
        handler=write_file,
        category="filesystem",
        capability_name="write_file",
        capability_description=(
            "Create or modify files on the local filesystem. "
            "External files are safely copied to a working area; originals are never touched."
        ),
        capability_refusal="I can't write files right now.",
    ))

    # ── patch_file ─────────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="patch_file",
        description=(
            "Make a precise in-place edit inside an existing file: replace "
            "old_string with new_string.\n\n"
            "WORKSPACE ROUTING: same as write_file — external files get a working "
            "copy in active/; the original is never modified directly.\n\n"
            "GUARANTEES:\n"
            "  • Fails clearly if old_string is not found — no silent no-ops.\n"
            "  • Fails if old_string matches more than once (unless replace_all=true), "
            "reporting every line number so you can make the string more specific.\n"
            "  • Near-miss detection: if the mismatch is only trailing whitespace, "
            "shows the exact file text so you can self-correct without re-reading.\n"
            "  • Preserves original encoding and line endings (LF/CRLF) exactly.\n"
            "  • Atomic write — power failure cannot corrupt the file.\n\n"
            "BEST PRACTICE: include 1–3 lines of surrounding context in old_string "
            "to make it unique. The more context, the safer the edit."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "Exact text to find and replace. Must appear exactly once "
                        "in the file (unless replace_all=true). Include surrounding "
                        "context lines to make it unique."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": (
                        "Text to substitute in place of old_string. "
                        "Pass an empty string to delete the match."
                    ),
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "Replace every occurrence of old_string (not just the first). "
                        "Default false — use only when all occurrences must change."
                    ),
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=patch_file,
        category="filesystem",
        capability_name="patch_file",
        capability_description=(
            "Make precise in-place edits to files without rewriting them. "
            "Preserves encoding, line endings, and protects against partial writes."
        ),
        capability_refusal="I can't edit files right now.",
    ))


# Auto-discovery alias
register = register_fs_tools
