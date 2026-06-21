"""
BRAIN/tools/_agent_tools.py — spawn_agent tool.

Registered explicitly by Brain._setup_core() (not auto-discovered, since
it needs the live tool_registry and llm refs as a closure at registration time).

Files starting with '_' are intentionally skipped by _auto_register_tools.
"""

import logging

_log = logging.getLogger("sofi.brain.tools.agents")


def register_agent_tools(registry, llm) -> None:
    """
    Register the spawn_agent tool. Called from Brain._setup_core() after
    the registry and LLM client are both initialized.
    """
    from BRAIN.agents.definitions import AGENT_DEFINITIONS
    from BRAIN.agents.orchestrator import run_subagent
    from BRAIN.tools.registry import ToolEntry

    agents_desc = "; ".join(
        f"'{name}' — {d['description']}"
        for name, d in AGENT_DEFINITIONS.items()
    )

    async def spawn_agent(agent_type: str, task: str) -> str:
        return await run_subagent(agent_type, task, registry, llm)

    registry.register(ToolEntry(
        name="spawn_agent",
        description=(
            "Spawn a specialized sub-agent to handle a complex multi-step task that would "
            "take many tool-call iterations if done directly. The sub-agent works independently "
            "using a restricted tool set and returns a structured summary of its findings or actions. "
            f"Available agents: {agents_desc}. "
            "Use for: deep research tasks, series of file edits, or any task needing 3+ tool calls."
        ),
        schema={
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": list(AGENT_DEFINITIONS.keys()),
                    "description": (
                        "Which sub-agent to spawn. "
                        + "; ".join(
                            f"'{k}': {v['description']}"
                            for k, v in AGENT_DEFINITIONS.items()
                        )
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Clear, complete description of what the sub-agent should do "
                        "and what result you need back. Include all relevant context — "
                        "the sub-agent has no memory of the current conversation."
                    ),
                },
            },
            "required": ["agent_type", "task"],
        },
        handler=spawn_agent,
        category="agents",
        capability_name="spawn_agent",
        capability_description="For research or file/code tasks needing 4+ tool calls, spawn a sub-agent — it runs them independently and returns a summary, faster than doing it turn by turn.",
        capability_refusal="I can't spawn sub-agents right now.",
    ))

    _log.debug(
        "register_agent_tools | spawn_agent registered | agents=%s",
        list(AGENT_DEFINITIONS.keys()),
    )
