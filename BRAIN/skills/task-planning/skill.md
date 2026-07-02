---
name: task-planning
description: >
  Break a complex or multi-step task into a visible plan before diving in. Trigger when a
  task has 3+ distinct steps, involves multiple tools or agents, or when the right approach
  is not obvious upfront. The rule: produce the plan AND immediately start step 1 in the
  same response — planning without executing is just structured procrastination.
requires: []
tags: [planning, orchestration, tasks, multi-step]
---

## Why plan before acting

For anything non-trivial, a visible plan prevents two common failure modes:
- Starting down a path and realising mid-way that the approach was wrong
- Doing the work but missing what was actually asked for

A plan also lets Zafar correct course before you have done the wrong work.

## Planning format

```
**Plan: [task in one line]**

Goal: [specific and measurable — what does success look like?]

Steps:
1. [Action] — Tool/agent: [what you will use]
2. [Action] — Tool/agent: [what you will use]
3. Deliver: [output format, length, where it lands]

Starting step 1...
```

Execute step 1 immediately after writing the plan in the same response.

## Choosing tools vs agents per step

| Step type | Use |
|---|---|
| Single file read or directory list | Direct tool: `read_file`, `list_directory` |
| Single web search or fetch | Direct tool: `web_search`, `web_fetch` |
| 4+ searches + source fetching | `spawn_agent` (research agent) — runs in background |
| Long-form document (500+ words) | `spawn_agent` (writer agent) — runs in background |
| Multi-file analysis + computation | `spawn_agent` (analyst agent) — runs in background |
| Multi-file edits with testing | `spawn_agent` (swe agent) — runs in background |

Use direct tools for simple, quick steps. Spawning an agent for a one-off file read adds
latency with no benefit.

## Background agent steps

When a step uses `spawn_agent`, it runs non-blocking — the plan continues when the agent
notifies completion via WHAT I'VE BEEN DOING. Plan for this:

```
Steps:
1. Spawn research agent with full brief — Background, ~2-5 min
2. [Continues when research completes] Spawn writer agent with findings — Background, ~1-2 min
3. [Continues when writer completes] Deliver report to Zafar
```

After spawning a background agent step, acknowledge and continue the conversation.
The plan resumes when the notification arrives.

## Execution rules

1. Follow the plan in order — if a step's result changes the approach, say so and adjust
2. After each step, verify: did this produce what the next step needs?
3. Final output is your inline response unless it is long-form — then it is a file via writer agent
4. For multi-turn tasks: end the turn with exactly what was done and what comes next

## When a plan is not needed

Skip the planning format for:
- Single-tool tasks ("read this file and summarise it")
- Quick lookups with a clear answer path
- Conversational requests that do not involve structured execution

Plan when the task is genuinely complex or when the wrong approach would waste significant effort.
