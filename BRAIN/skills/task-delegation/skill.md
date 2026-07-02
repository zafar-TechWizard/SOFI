---
name: task-delegation
description: >
  How to delegate work to background sub-agents — write a focused task brief, choose the
  right agent type, and process the result when it returns. Load this before any spawn_agent
  call. Trigger when a task needs 4+ tool calls, produces long-form output, or can usefully
  run in parallel while the conversation continues. A vague brief produces vague output —
  this skill teaches how to write briefs that succeed.
requires: [spawn_agent, read_file]
tags: [agents, delegation, workflow]
---

## The golden rule

**The sub-agent's only context is the task string you write.** It has no memory of this
conversation, no persona, no access to Zafar's history. Everything it needs — WHAT to do,
HOW to approach it, what OUTPUT FORMAT to produce, what LENGTH to target, and any relevant
CONTEXT — must be in the task brief. A vague brief produces vague output.

## Step 1 — Acknowledge before delegating

Always say what you are doing before calling spawn_agent. One line:
- "Running deep research in background."
- "Delegating to the analyst — should have results shortly."
- "Writing that up now, I'll surface it when done."

Never call spawn_agent silently. The acknowledgement is not optional.

## Step 2 — Choose the right agent

| Agent | Use for |
|---|---|
| `research` | Web research, gathering information from multiple sources |
| `writer` | Long-form documents, reports, analyses, structured content |
| `analyst` | Code analysis, data examination, file inspection, computation |
| `planner` | Breaking down complex tasks, building execution plans |
| `code` | Targeted file edits, bug fixes, single-feature additions |
| `swe` | Full software engineering: read -> plan -> implement -> test |

## Step 3 — Write a complete task brief

Every brief needs four parts — all four, always:

**WHAT** — State the task explicitly, not the vague intent.
> "Research the current state of vector databases as of 2025, covering top options, performance benchmarks, and use cases."
> NOT: "Research vector databases."

**HOW** — Tell the agent what approach to take.
> "Search for recent benchmarks, compare Pinecone, Weaviate, Qdrant, Chroma. Fetch 2-3 authoritative sources."
> NOT: (omitted — the agent will guess, usually badly)

**OUTPUT FORMAT** — Specify structure explicitly.
> "Return: Executive Summary, Comparison Table (performance / ecosystem / cost), Top 3 Recommendations, When to use each."
> NOT: "Return your findings."

**LENGTH** — State a word target and say do not truncate.
> "Target 1500-2000 words. Do not truncate — deliver complete content."
> NOT: (omitted — agent defaults to minimal output)

End every task brief with `INFORM: true` on its own line to be notified when it completes.

## Step 4 — After delegating

Give Zafar a brief acknowledgement and continue the conversation normally:
- "On it — research running in background. I'll surface the findings when done."
- "Writer agent is on it. Result shortly."

Do not ask Zafar to wait or pause. The agent runs independently.

## Step 5 — When the result arrives

When WHAT I'VE BEEN DOING shows a COMPLETED task:

1. Read the file: `read_file` on the `result_path`
2. Understand the original request — what did Zafar actually want?
3. Act based on that:
   - Question asked -> answer it directly using the findings
   - Report requested -> spawn a writer agent to format it properly
   - Analysis wanted -> synthesize into a clear recommendation
   - Summary requested -> 2-3 sentence synthesis in your response

Never dump raw agent output inline. Always process it — read, understand, then respond.

## Good vs bad brief

**Bad** (too vague — the agent cannot succeed):
```
task="Research React vs Vue for the project."
```

**Good** (complete — the agent has everything it needs):
```
task="""Compare React vs Vue.js for a new web dashboard project.

Context: ~10 developers, mixed JavaScript experience, 2-year lifespan, medium complexity.

Approach: Search for recent (2024-2025) comparisons, benchmark data, developer survey
results, ecosystem health, and hiring market data.

Output format:
## React vs Vue: Decision Brief
**Executive Summary** (3-4 sentences)
**Comparison Table** (performance, ecosystem, learning curve, job market, tooling)
**Recommendation** with reasoning
**When to reconsider**

Target: 800-1000 words. Specific data, not generalities.
INFORM: true"""
```
