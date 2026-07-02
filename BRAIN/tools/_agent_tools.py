"""
BRAIN/tools/_agent_tools.py — spawn_agent tool (production version).

Uses SubAgentRunner for the agentic loop, ActiveRegistry for concurrency
control, and SubAgentResult for structured output. Supports batch mode.

Files starting with '_' are intentionally skipped by _auto_register_tools.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Callable, Optional

from BRAIN.agents.registry import ActiveRegistry, SubAgentRecord

_log = logging.getLogger("sofi.brain.tools.agents")

_running_agent_tasks: set = set()


def register_agent_tools(
    registry,
    llm,
    get_workspace: Optional[Callable] = None,
    register_bg_task: Optional[Callable] = None,
    active_registry: Optional[ActiveRegistry] = None,
) -> None:
    from BRAIN.agents.definitions import AGENT_DEFINITIONS
    from BRAIN.tasks.task_manager import TaskManager
    from BRAIN.tools.registry import ToolEntry

    task_manager = TaskManager()

    agents_desc = "; ".join(
        f"'{name}' — {d['description']}"
        for name, d in AGENT_DEFINITIONS.items()
    )

    async def spawn_agent(agent_type: str, task: str) -> str:
        from BRAIN.agents.runner import SubAgentRunner

        if agent_type not in AGENT_DEFINITIONS:
            available = ", ".join(AGENT_DEFINITIONS.keys())
            return f"Unknown agent '{agent_type}'. Available: {available}"

        # Concurrency check
        if active_registry and not active_registry.has_capacity():
            snap = active_registry.snapshot()
            active_names = [a["agent_type"] for a in snap["active"]]
            return (
                f"All {active_registry.max_concurrent} agent slots are in use "
                f"({', '.join(active_names)}). Wait for one to finish or interrupt one first."
            )

        task_file = task_manager.create_task(
            original_query=task[:200],
            brief=task,
            agent_type=agent_type,
        )
        task_id = task_file.task_id

        _log.info(
            "spawn_agent | task created | id=%s agent=%s brief_len=%d",
            task_id, agent_type, len(task),
        )

        ws = get_workspace() if get_workspace else None
        if ws:
            from memory.working_memory.working_context import (
                WorkspaceItem,
                WorkspaceItemStatus,
                WorkspaceItemType,
            )
            ws.add_item(WorkspaceItem(
                id=task_id,
                type=WorkspaceItemType.TASK,
                title=f"{agent_type}: {task[:60]}",
                description=task[:200],
                status=WorkspaceItemStatus.IN_PROGRESS,
                progress=0.0,
                source_agent=agent_type,
                metadata={"task_id": task_id, "agent_type": agent_type},
            ))

        runner = SubAgentRunner(
            agent_name=agent_type,
            task_brief=task,
            tool_registry=registry,
            llm=llm,
            task_manager=task_manager,
            task_id=task_id,
            registry=active_registry,
        )
        subagent_id = runner.subagent_id

        # Register in active registry
        if active_registry:
            record = SubAgentRecord(
                subagent_id=subagent_id,
                task_id=task_id,
                agent_type=agent_type,
                query=task[:200],
            )
            if not active_registry.register(record):
                task_manager.mark_failed(task_id, "Registry full — could not register")
                return "Internal error: agent registry full."

        async def _run():
            try:
                result = await runner.run()

                # Write delivery from structured result
                delivery = result.to_delivery()
                task_manager.write_delivery(
                    task_id=task_id,
                    status=delivery["status"],
                    summary=delivery["summary"],
                    content=delivery["content"],
                    gaps=delivery.get("gaps"),
                )

                # Store metrics on the task
                task_manager.update_metadata(task_id, {
                    "metrics": result.metrics_dict(),
                    "files_written": result.files_written,
                })

                _log.info(
                    "spawn_agent | delivery written | id=%s agent=%s "
                    "status=%s exit=%s duration=%.1fs iters=%d content_len=%d",
                    task_id, agent_type, result.status, result.exit_reason,
                    result.duration_seconds, result.iterations,
                    len(result.content),
                )

                # Update workspace
                if ws:
                    from memory.working_memory.working_context import (
                        NotifyPriority,
                        WorkspaceItemStatus,
                    )
                    ws_status = (
                        WorkspaceItemStatus.COMPLETED
                        if result.status == "completed"
                        else WorkspaceItemStatus.FAILED
                    )
                    ws.update_item(
                        task_id,
                        status=ws_status,
                        progress=1.0,
                        notify=True,
                        notify_priority=NotifyPriority.NORMAL,
                        description=f"Delivery ready: {result.summary[:200]}",
                        metadata={
                            "task_id": task_id,
                            "agent_type": agent_type,
                            "completed_at": datetime.now().isoformat(),
                            "content_len": len(result.content),
                            "metrics": result.metrics_dict(),
                        },
                    )

                # File conflict detection
                if result.file_conflicts:
                    _log.warning(
                        "spawn_agent | file conflicts | id=%s conflicts=%s",
                        task_id, result.file_conflicts,
                    )

            except Exception as exc:
                _log.error(
                    "spawn_agent | failed | id=%s agent=%s err=%s",
                    task_id, agent_type, exc, exc_info=True,
                )
                task_manager.mark_failed(task_id, str(exc))

                if ws:
                    from memory.working_memory.working_context import WorkspaceItemStatus
                    ws.update_item(
                        task_id,
                        status=WorkspaceItemStatus.FAILED,
                        notify=True,
                        description=f"Failed: {exc}",
                    )
            finally:
                if active_registry:
                    active_registry.unregister(subagent_id)

        bg_task = asyncio.create_task(
            _run(), name=f"agent-{agent_type}-{task_id}"
        )
        if active_registry:
            rec = active_registry.get(subagent_id)
            if rec:
                rec._task = bg_task
        _running_agent_tasks.add(bg_task)
        bg_task.add_done_callback(_running_agent_tasks.discard)

        if register_bg_task:
            register_bg_task(bg_task)

        return (
            f"Internal {agent_type} process started (task {task_id}). "
            f"Working on it — I'll have the results shortly."
        )

    registry.register(ToolEntry(
        name="spawn_agent",
        description=(
            "Delegate a task to one of my internal processes. The process runs in "
            "the background — I stay available for conversation. When it completes, "
            "the delivery appears in my context and I deliver the results.\n\n"
            "HOW TO USE:\n"
            "1. Write a DETAILED brief as the task parameter — this is the ONLY context "
            "the internal process receives. Include: what to do, approach, output format, "
            "expected length, success criteria, and ALL relevant context.\n"
            "2. After spawning, acknowledge to Zafar what I'm working on.\n"
            "3. When the delivery appears in COMPLETED DELIVERIES, read the content "
            "and deliver it to Zafar in my own voice.\n\n"
            "IMPORTANT: The internal process does NOT know who I'm talking to. It only "
            "knows the task brief I give it. Write the brief from my perspective — "
            "'I need to find...' not 'The user wants...'\n\n"
            f"Available processes: {agents_desc}."
        ),
        schema={
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": list(AGENT_DEFINITIONS.keys()),
                    "description": (
                        "Which internal process to run. "
                        + "; ".join(
                            f"'{k}': {v['description']}"
                            for k, v in AGENT_DEFINITIONS.items()
                        )
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Complete task brief for the internal process. Write from SOFi's "
                        "perspective ('I need to find...'). Must include: WHAT to do, "
                        "HOW to approach it, expected OUTPUT FORMAT and length, SUCCESS "
                        "CRITERIA, and all CONTEXT needed. The process has no other context."
                    ),
                },
            },
            "required": ["agent_type", "task"],
        },
        handler=spawn_agent,
        timeout=30.0,
        category="agents",
        capability_name="spawn_agent",
        capability_description=(
            "Delegate focused work to internal processes (research, writing, "
            "analysis, coding). Runs in background; delivery written to disk."
        ),
        capability_refusal="Internal processes unavailable right now.",
    ))

    registry._task_manager = task_manager

    _log.debug(
        "register_agent_tools | spawn_agent registered | agents=%s",
        list(AGENT_DEFINITIONS.keys()),
    )
