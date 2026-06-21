"""
BRAIN/tools/skill_tools.py — Skills discovery and loading tools.

Auto-discovered by _auto_register_tools. These tools let SOFi look up
and load her own skill playbooks at runtime.

- skills_list : returns all available skills with names and descriptions
- skills_load : loads the full instructions for a named skill
"""

import logging

from BRAIN.skills._registry import registry as _skill_registry
from BRAIN.tools.registry import ToolEntry

_log = logging.getLogger("sofi.brain.tools.skills")


async def skills_list() -> str:
    """Return a formatted list of all available skills."""
    skills = _skill_registry.list()
    if not skills:
        return "No skills available."

    lines = ["Available skills:\n"]
    for s in skills:
        req = f"  requires: {', '.join(s.requires)}" if s.requires else ""
        tags = f"  tags: {', '.join(s.tags)}" if s.tags else ""
        lines.append(f"• {s.name} — {s.description}")
        if req:
            lines.append(req)
        if tags:
            lines.append(tags)
    lines.append(f"\nTotal: {len(skills)} skill(s). Use skills_load(skill_name) to get full instructions.")
    return "\n".join(lines)


async def skills_load(skill_name: str) -> str:
    """Return the full instructions for a named skill."""
    skill = _skill_registry.get(skill_name)
    if skill is None:
        available = ", ".join(_skill_registry.names()) or "none"
        return f"Skill '{skill_name}' not found. Available: {available}"

    return (
        f"# Skill: {skill.title}\n\n"
        f"{skill.content}\n\n"
        f"---\n"
        f"*Requires: {', '.join(skill.requires) if skill.requires else 'none'}*"
    )


def register_skill_tools(registry) -> None:
    registry.register(ToolEntry(
        name="skills_list",
        description=(
            "List all available skills (built-in playbooks for specific tasks). "
            "Returns each skill's name, description, and required tools. "
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
        capability_description="Discover available built-in skill playbooks.",
        capability_refusal="I can't list skills right now.",
    ))

    registry.register(ToolEntry(
        name="skills_load",
        description=(
            "Load the full step-by-step instructions for a specific skill by name. "
            "Use after skills_list to get the complete playbook before executing a skill. "
            "The instructions tell you exactly how to approach the task using your tools."
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
        capability_description="Load detailed instructions for a specific built-in skill.",
        capability_refusal="I can't load skills right now.",
    ))

    _log.debug(
        "register_skill_tools | registered | skills=%s",
        _skill_registry.names(),
    )


register = register_skill_tools
