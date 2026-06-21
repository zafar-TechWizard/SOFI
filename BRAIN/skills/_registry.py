"""
BRAIN/skills/_registry.py — Skill definition registry.

Scans BRAIN/skills/*.md files at first access, parses their YAML frontmatter,
and makes them available as queryable Skill objects.

Skill files have the format:

    ---
    name: daily_briefing
    title: Daily Briefing
    description: Compile and deliver a morning summary
    requires: [web_search, get_weather]
    tags: [productivity, morning]
    ---

    # Full skill instructions as markdown...

The frontmatter parser handles flat key:value pairs and simple [list] syntax.
No PyYAML dependency.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger("sofi.brain.skills")

_SKILLS_DIR = Path(__file__).parent


@dataclass
class Skill:
    name: str
    title: str
    description: str
    content: str
    requires: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-style frontmatter from a markdown string."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text

    fm_lines = lines[1:end]
    body = "\n".join(lines[end + 1:]).strip()

    meta: dict = {}
    for line in fm_lines:
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            val = [v.strip() for v in inner.split(",") if v.strip()]  # type: ignore[assignment]
        meta[key] = val

    return meta, body


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load()

    def _load(self) -> None:
        count = 0
        for path in sorted(_SKILLS_DIR.glob("*.md")):
            if path.stem.startswith("_"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(text)
                if not meta.get("name"):
                    _log.warning("skill file %s has no 'name' field — skipped", path.name)
                    continue
                skill = Skill(
                    name=str(meta.get("name", path.stem)),
                    title=str(meta.get("title", path.stem)),
                    description=str(meta.get("description", "")),
                    content=body,
                    requires=meta.get("requires", []) if isinstance(meta.get("requires"), list) else [],
                    tags=meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
                )
                self._skills[skill.name] = skill
                count += 1
                _log.debug("skill loaded | name=%s title=%s", skill.name, skill.title)
            except Exception as exc:
                _log.warning("skill load error | file=%s exc=%s", path.name, exc)

        _log.info("skills | loaded %d skill(s) from %s", count, _SKILLS_DIR)

    def list(self) -> List[Skill]:
        self._ensure_loaded()
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Optional[Skill]:
        self._ensure_loaded()
        return self._skills.get(name)

    def names(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._skills.keys())

    def reload(self) -> None:
        self._skills.clear()
        self._loaded = False
        self._ensure_loaded()


# Module-level singleton — imported by skill_tools.py and brain.py diagnostics.
registry = SkillRegistry()
