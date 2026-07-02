"""
BRAIN/tools/skill_tools.py — Skills discovery and loading tools.

Auto-discovered by _auto_register_tools. These tools let SOFi look up
and load her own skill playbooks at runtime.

- skills_list : returns all available skills with names and descriptions
- skills_load : loads the full instructions for a named skill; enforces
                skill.requires — refuses with a clear message if any
                required tool is missing or unavailable, so SOFi never
                starts a skill it can't complete.
"""

import logging

from BRAIN.skills._registry import registry as _skill_registry
from BRAIN.tools.registry import ToolEntry

_log = logging.getLogger("sofi.brain.tools.skills")


def register_skill_tools(tool_registry) -> None:

    async def skills_list() -> str:
        """Return a formatted list of all available skills."""
        skills = _skill_registry.list()
        if not skills:
            return "No skills available."

        lines = ["Available skills:\n"]
        for s in skills:
            # Flag whether skill is immediately runnable given current tools
            if s.requires:
                missing = _missing_tools(s.requires, tool_registry)
                status = "⚠ missing tools" if missing else "✓ ready"
                req_line = f"  requires: {', '.join(s.requires)} [{status}]"
            else:
                req_line = ""
            tags_line = f"  tags: {', '.join(s.tags)}" if s.tags else ""

            lines.append(f"• {s.name} — {s.description}")
            if req_line:
                lines.append(req_line)
            if tags_line:
                lines.append(tags_line)

        lines.append(
            f"\nTotal: {len(skills)} skill(s). "
            "Use skills_load(skill_name) to get full instructions."
        )
        return "\n".join(lines)

    async def skills_load(skill_name: str) -> str:
        """Return the full instructions for a named skill, if all required tools are present."""
        skill = _skill_registry.get(skill_name)
        if skill is None:
            available = ", ".join(_skill_registry.names()) or "none"
            return f"Skill '{skill_name}' not found. Available: {available}"

        # Enforcement gate: check required tools before serving the playbook.
        if skill.requires:
            missing = _missing_tools(skill.requires, tool_registry)
            if missing:
                return (
                    f"Skill '{skill.name}' cannot run — the following required tools "
                    f"are not registered or unavailable: {', '.join(missing)}.\n"
                    f"All required tools: {', '.join(skill.requires)}.\n"
                    f"Add the missing tool files and restart or /reload before using this skill."
                )

        return (
            f"# Skill: {skill.title}\n\n"
            f"{skill.content}\n\n"
            f"---\n"
            f"*Requires: {', '.join(skill.requires) if skill.requires else 'none'}*"
        )

    tool_registry.register(ToolEntry(
        name="skills_list",
        description=(
            "List all available skills (built-in playbooks for specific tasks). "
            "Returns each skill's name, description, required tools, and readiness status. "
            "Use this to discover what structured playbooks are available before running a complex task."
        ),
        schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=skills_list,
        category="skills",
        capability_name="skills_list",
        capability_description="Before any structured multi-step task (briefing, code review, research), check if a skill playbook exists for it.",
        capability_refusal="I can't list skills right now.",
    ))

    tool_registry.register(ToolEntry(
        name="skills_load",
        description=(
            "Load the full step-by-step instructions for a specific skill by name. "
            "Use after skills_list to get the complete playbook before executing a skill. "
            "Returns an error if any required tool is missing — skill won't load until all "
            "required tools are available."
        ),
        schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load (from skills_list output)",
                },
            },
            "required": ["skill_name"],
        },
        handler=skills_load,
        category="skills",
        capability_name="skills_load",
        capability_description="Load the step-by-step playbook for a skill found via skills_list, then follow it.",
        capability_refusal="I can't load skills right now.",
    ))

    _log.debug(
        "register_skill_tools | registered | skills=%s",
        _skill_registry.names(),
    )


def _missing_tools(requires: list, tool_registry) -> list:
    """Return tool names from requires that are not registered or not available."""
    missing = []
    for tool_name in requires:
        entry = tool_registry.get(tool_name)
        if entry is None or not entry.is_available():
            missing.append(tool_name)
    return missing


register = register_skill_tools
