"""
BRAIN/agents/orchestrator.py — Sub-agent runner.

run_subagent(name, task, tool_registry, llm, task_manager, task_id) drives
a mini agentic loop for a named sub-agent. The sub-agent:
  - Gets its role instructions as the system prompt
  - Gets the task brief as the user prompt
  - Has access to role-specific tools + update_task_progress
  - Updates progress on disk via TaskManager throughout
  - Returns plain-text findings for SOFi to synthesize

The task brief (user prompt) contains: what to do, approach, output format,
success criteria, and all context. The system prompt carries identity + role.
"""

import asyncio
import json
import logging
from typing import Optional

from BRAIN.agents.definitions import AGENT_DEFINITIONS

_log = logging.getLogger("sofi.brain.agents")

_NEVER_ALLOW = frozenset({"spawn_agent", "spawn_agent_background"})


async def run_subagent(
    name: str,
    task: str,
    tool_registry,
    llm,
    task_manager=None,
    task_id: str = "",
    max_iterations: int | None = None,
) -> str:
    """
    Run a named sub-agent to completion. Returns its text output.

    The sub-agent runs a private agentic loop (no persona, no memory)
    using only the tools listed in its definition + update_task_progress.

    System prompt = identity preamble + role instructions (from definitions.py)
    User prompt   = the task brief (from SOFi)
    """
    defn = AGENT_DEFINITIONS.get(name)
    if defn is None:
        available = ", ".join(AGENT_DEFINITIONS.keys())
        return f"[spawn_agent] Unknown agent '{name}'. Available: {available}"

    allowed: set[str] = set(defn["tools"]) - _NEVER_ALLOW
    system: str = defn["system"]
    max_iters: int = max_iterations or defn.get("max_iterations", 6)

    # Build the tool definitions available to this agent.
    all_defs = tool_registry.get_definitions()
    agent_tool_defs = [
        d for d in all_defs
        if d["function"]["name"] in allowed
    ]

    # Inject the update_task_progress tool — this is how the sub-agent
    # communicates progress back to SOFi via disk-backed task files.
    if task_manager and task_id:
        agent_tool_defs.append(_progress_tool_def(task_id))

    _log.info(
        "subagent | start | name=%s tools=%d task_preview=%.60s",
        name, len(agent_tool_defs), task,
    )

    # System prompt = identity + role instructions
    # User prompt = the task brief from SOFi
    messages = [{"role": "user", "content": task}]
    output = ""

    if task_manager and task_id:
        task_manager.update_status(task_id, "in_progress")

    for iteration in range(1, max_iters + 1):
        _log.debug("subagent %s | iter=%d", name, iteration)

        try:
            response = await llm.call_with_tools(system, messages, agent_tool_defs)
        except Exception as exc:
            _log.warning("subagent %s iter=%d | LLM error: %s", name, iteration, exc)
            if task_manager and task_id:
                task_manager.mark_failed(task_id, f"LLM error: {exc}")
            return f"[sub-agent {name} hit an error: {exc}]"

        if response.tool_calls:
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

            from BRAIN.tools.registry import ToolCall

            async def _exec(tc):
                # Handle update_task_progress internally
                if tc.name == "update_task_progress":
                    return tc.id, _handle_progress_update(
                        tc.arguments, task_manager, task_id
                    )
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

    # Self-verification status update
    if task_manager and task_id:
        task_manager.update_status(task_id, "verifying")

    return output


def _progress_tool_def(task_id: str) -> dict:
    """Build the update_task_progress tool definition for sub-agents."""
    return {
        "type": "function",
        "function": {
            "name": "update_task_progress",
            "description": (
                "Report your progress on the current task. Call this after each "
                "major step to keep SOFi informed. Include what you just did, "
                "what you found, and what you're doing next."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step_description": {
                        "type": "string",
                        "description": "What you just completed (e.g., 'Searched web for quantum computing 2026')",
                    },
                    "findings": {
                        "type": "string",
                        "description": "Key findings from this step (brief — full details go in your final output)",
                    },
                    "next_action": {
                        "type": "string",
                        "description": "What you're doing next (e.g., 'Fetching full article from source #2')",
                    },
                    "steps_plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Your full step plan (only include on first call to set the plan). "
                            "Each item is a short action description."
                        ),
                    },
                },
                "required": ["step_description", "findings"],
            },
        },
    }


def _handle_progress_update(
    args: dict,
    task_manager,
    task_id: str,
) -> str:
    """Handle the update_task_progress tool call from a sub-agent."""
    if not task_manager or not task_id:
        return "Progress noted."

    # First call with steps_plan: set the task steps
    steps_plan = args.get("steps_plan")
    if steps_plan:
        steps = [
            {"step": i + 1, "action": action, "status": "pending", "detail": ""}
            for i, action in enumerate(steps_plan)
        ]
        task_manager.set_steps(task_id, steps)

    # Update current step
    task = task_manager.get_task(task_id)
    if task and task.steps:
        # Find first non-done step and mark it
        for i, step in enumerate(task.steps):
            if step.get("status") in ("pending", "in_progress"):
                detail = args.get("findings", "")
                task_manager.update_step(task_id, i, "done", detail)
                break

    _log.debug(
        "progress_update | task=%s step=%s findings_len=%d",
        task_id,
        args.get("step_description", "?")[:40],
        len(args.get("findings", "")),
    )

    next_action = args.get("next_action", "")
    if next_action:
        return f"Progress recorded. Continue with: {next_action}"
    return "Progress recorded."
