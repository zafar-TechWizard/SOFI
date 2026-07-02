"""
BRAIN/skills/_registry.py — Skill definition registry.

Scans skill subdirectories at first access. Each skill lives in its own folder:

    BRAIN/skills/code-review/
        skill.md       ← required: playbook and instructions
        agent.md       ← optional: custom agent definition for this skill

    SKILLS/code-review/   ← user drop-in directory (workspace root)
        skill.md
        agent.md       ← optional

Skill folders are discovered from two roots:
  1. BRAIN/skills/     — built-in skills (shipped with code)
  2. SKILLS/           — user drop-in skills (add any folder here, no code changes)

User skills override built-in skills with the same name.
Folders starting with '_' or '.' are skipped.
The folder name is used as the default skill name if frontmatter has no 'name'.

skill.md frontmatter format:

    ---
    name: code_review             # defaults to folder name (hyphens → underscores)
    title: Code Review
    description: Review code for bugs and issues
    requires: [read_file, search_files]
    tags: [code, review]
    ---

    # Full playbook as markdown...

agent.md (optional) — custom agent instructions SOFi can inject when spawning
an agent for this skill. Loaded alongside the skill; accessible via skill.agent_prompt.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger("sofi.brain.skills")

# Built-in skills live in BRAIN/skills/<skill-name>/skill.md
_SKILLS_DIR = Path(__file__).parent

# User drop-in skills live in SKILLS/<skill-name>/skill.md at workspace root
_USER_SKILLS_DIR = Path(__file__).parent.parent.parent / "SKILLS"


@dataclass
class Skill:
    name: str
    title: str
    description: str
    content: str                                            # skill.md body
    requires: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    agent_prompt: Optional[str] = field(default=None)      # agent.md body if present
    folder: Optional[Path] = field(default=None, compare=False)  # path to skill folder


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-style frontmatter. Handles flat key:value and [list] values."""
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


def _folder_to_name(folder_name: str) -> str:
    """Convert folder name to a valid skill name: hyphens/spaces → underscores."""
    return folder_name.replace("-", "_").replace(" ", "_").lower()


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load()

    def _load_skill_folder(self, folder: Path, source_label: str) -> Optional[Skill]:
        """
        Load a skill from a folder. Returns the Skill or None if skipped.

        Looks for skill.md as the primary file. Falls back to any .md file
        in the folder if skill.md doesn't exist. Also loads agent.md if present.
        """
        # Find primary skill file
        skill_file = folder / "skill.md"
        if not skill_file.exists():
            # Fall back to any .md file (alphabetical — first one wins)
            candidates = sorted(
                f for f in folder.glob("*.md")
                if not f.stem.startswith("_") and f.stem != "agent"
            )
            if not candidates:
                _log.debug("skill folder %s has no skill.md — skipped", folder.name)
                return None
            skill_file = candidates[0]

        try:
            text = skill_file.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)

            # Use frontmatter name, or derive from folder name
            name = str(meta.get("name", _folder_to_name(folder.name)))

            # Load optional agent.md
            agent_prompt: Optional[str] = None
            agent_file = folder / "agent.md"
            if agent_file.exists():
                try:
                    agent_text = agent_file.read_text(encoding="utf-8")
                    _, agent_body = _parse_frontmatter(agent_text)
                    agent_prompt = agent_body.strip() or None
                except Exception as exc:
                    _log.warning("agent.md load error | folder=%s exc=%s", folder.name, exc)

            skill = Skill(
                name=name,
                title=str(meta.get("title", folder.name)),
                description=str(meta.get("description", "")),
                content=body,
                requires=(
                    meta.get("requires", [])
                    if isinstance(meta.get("requires"), list) else []
                ),
                tags=(
                    meta.get("tags", [])
                    if isinstance(meta.get("tags"), list) else []
                ),
                agent_prompt=agent_prompt,
                folder=folder,
            )
            _log.debug(
                "skill loaded | name=%s source=%s/%s agent=%s",
                name, source_label, folder.name, "yes" if agent_prompt else "no",
            )
            return skill

        except Exception as exc:
            _log.warning("skill load error | folder=%s exc=%s", folder.name, exc)
            return None

    def _load_dir(self, directory: Path, label: str) -> int:
        """
        Scan a root directory for skill subfolders. Each subfolder = one skill.
        Returns count of skills loaded.
        """
        if not directory.exists():
            return 0

        count = 0
        for entry in sorted(directory.iterdir()):
            # Skills must be in subfolders, not flat files
            if not entry.is_dir():
                continue
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue

            skill = self._load_skill_folder(entry, label)
            if skill is None:
                continue

            existed = skill.name in self._skills
            self._skills[skill.name] = skill
            count += 1
            if existed:
                _log.debug("skill override | name=%s by=%s", skill.name, label)

        return count

    def _load(self) -> None:
        builtin = self._load_dir(_SKILLS_DIR, "builtin")
        user = self._load_dir(_USER_SKILLS_DIR, "user")
        total = builtin + user
        _log.info(
            "skills | loaded %d skill(s) | builtin=%d user-drop-in=%d",
            total, builtin, user,
        )

    def list(self) -> List[Skill]:
        self._ensure_loaded()
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Optional[Skill]:
        self._ensure_loaded()
        return self._skills.get(name)

    def names(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._skills.keys())

    def list_available(self) -> List[tuple]:
        """Return (name, description) pairs for all loaded skills."""
        self._ensure_loaded()
        return [
            (s.name, s.description or s.title)
            for s in sorted(self._skills.values(), key=lambda s: s.name)
            if s.description or s.title
        ]

    def reload(self) -> None:
        self._skills.clear()
        self._loaded = False
        self._ensure_loaded()


# Module-level singleton — imported by skill_tools.py and brain.py diagnostics.
registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    """Return the module-level singleton."""
    return registry
