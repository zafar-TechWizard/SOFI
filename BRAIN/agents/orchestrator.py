"""
BRAIN/agents/orchestrator.py — Sub-agent runner.

run_subagent(name, task, tool_registry, llm) drives a mini agentic loop
for a named sub-agent. The sub-agent has no persona — just a task-focused
system prompt and a restricted tool set.

Returns: plain-text summary of the sub-agent's findings or actions.
The caller (brain.py via spawn_agent tool) synthesizes this into SOFi's voice.
"""

import asyncio
import json
import logging

from BRAIN.agents.definitions import AGENT_DEFINITIONS

_log = logging.getLogger("sofi.brain.agents")

# Recursive spawning guard: sub-agents may never call spawn_agent themselves.
_NEVER_ALLOW = frozenset({"spawn_agent"})


async def run_subagent(
    name: str,
    task: str,
    tool_registry,
    llm,
    max_iterations: int | None = None,
) -> str:
    """
    Run a named sub-agent to completion. Returns its text output.

    The sub-agent runs a private agentic loop (no persona, no memory)
    using only the tools listed in its definition. SOFi is not visible
    to or from the sub-agent — it sees only its task and tool results.
    """
    defn = AGENT_DEFINITIONS.get(name)
    if defn is None:
        available = ", ".join(AGENT_DEFINITIONS.keys())
        return f"[spawn_agent] Unknown agent '{name}'. Available: {available}"

    allowed: set[str] = set(defn["tools"]) - _NEVER_ALLOW
    system: str = defn["system"]
    max_iters: int = max_iterations or defn.get("max_iterations", 6)

    # Filter the full registry to only this agent's allowed tools
    all_defs = tool_registry.get_definitions()
    agent_tool_defs = [
        d for d in all_defs
        if d["function"]["name"] in allowed
    ]

    _log.info(
        "subagent | start | name=%s tools=%d task_preview=%.60s",
        name, len(agent_tool_defs), task,
    )

    messages = [{"role": "user", "content": task}]
    output = ""

    for iteration in range(1, max_iters + 1):
        _log.debug("subagent %s | iter=%d", name, iteration)

        try:
            response = await llm.call_with_tools(system, messages, agent_tool_defs)
        except Exception as exc:
            _log.warning("subagent %s iter=%d | LLM error: %s", name, iteration, exc)
            return f"[sub-agent {name} hit an error: {exc}]"

        if response.tool_calls:
            # Build the assistant message (preserving raw_content for Gemini
            # thinking models that embed thought_signature in tool call parts)
            assistant_msg: dict = {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            if response.raw_content is not None:
                assistant_msg["_raw_content"] = response.raw_content
            messages.append(assistant_msg)

            # Execute all tool calls concurrently, but only those in allowed set
            from BRAIN.tools.registry import ToolCall

            async def _exec(tc):
                if tc.name not in allowed:
                    return tc.id, f"Tool '{tc.name}' is not available to this agent."
                result = await tool_registry.execute(
                    ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                )
                _log.debug(
                    "subagent %s | tool=%s success=%s",
                    name, tc.name, result.success,
                )
                return tc.id, result.to_string()

            results = await asyncio.gather(
                *[_exec(tc) for tc in response.tool_calls],
                return_exceptions=True,
            )

            for tc, result_pair in zip(response.tool_calls, results):
                if isinstance(result_pair, Exception):
                    content = f"Tool error: {result_pair}"
                else:
                    _, content = result_pair
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })
            continue

        if response.text:
            output = response.text
            _log.info(
                "subagent %s | done after %d iter(s) | output_len=%d",
                name, iteration, len(output),
            )
            break

    if not output:
        output = (
            f"[Sub-agent '{name}' reached {max_iters} iterations without a final response. "
            "Partial results may be in the tool call history.]"
        )
        _log.warning("subagent %s | max_iterations=%d hit with no output", name, max_iters)

    return output
