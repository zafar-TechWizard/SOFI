---
name: deep-research
description: >
  Full research-to-report pipeline: spawn a research agent to gather multi-source findings,
  then a writer agent to produce a polished document saved to file. Trigger when Zafar says
  "research this", "write a report on", "deep dive on", "comprehensive overview of", or gives
  a topic alongside a word or token count. Use this — not research_deep — when the output is
  a formal report, not an inline answer.
requires: [spawn_agent, read_file]
tags: [research, writing, agents, orchestration]
---

## Why two agents

Research gathering and report writing are separate cognitive tasks. The research agent
focuses on finding and extracting — it is not concerned with prose. The writer agent turns
those findings into polished structured output at the requested length, without being
distracted by searching. Splitting roles produces sharper output than one agent doing both.

## Phase 1 — Kick off research

Acknowledge before spawning: "Deep research running in background — I'll surface the report when it's done."

Spawn a `research` agent. Write the full brief — the agent has no memory of this conversation:

```
task="""Research [TOPIC] thoroughly for a report.

Steps:
1. Broad search first — map the landscape, note the 3-5 most relevant results
2. Targeted follow-ups on: latest developments, main competing views, known pitfalls or controversies
3. Fetch 2-3 authoritative sources in full
4. Flag any contradictions between sources

Output structured findings:
- What it is (scope, definition)
- Current state as of 2025
- Key facts with specifics (numbers, names, dates)
- Main tradeoffs or competing views
- Sources (domain + title for each)

Target 800-1200 words of complete, detailed findings. Do not truncate.
The writer agent depends on your findings — give it everything.
INFORM: true"""
```

After spawning: respond in 1-2 sentences, then continue the conversation normally.

## Phase 2 — When research completes

When the research task appears in WHAT I'VE BEEN DOING -> COMPLETED:

1. Call `read_file` on the `result_path`
2. Note the original request: length, specific angle, format asked for
3. Spawn a `writer` agent immediately:

```
task="""Write a research report on [TOPIC]. Target: [LENGTH] words.

Source material (use this as your primary reference — do not search further):
---
[PASTE FULL RESEARCH FINDINGS HERE]
---

Structure:
# [Descriptive Title — specific, not generic]

**Summary:** 2-3 sentences capturing the bottom line

---

## Background
## [Core Section — the main substance, renamed to match the topic]
## Key Findings
## Considerations / Tradeoffs
## Conclusion

Rules:
- Begin with the title — no "Here is the report" preamble
- Bold the most important claim in each section
- Cite source context inline: "according to [source]..." or "as of 2025..."
- Every paragraph carries information — no padding
- Do not truncate. Deliver complete content at the requested length.
INFORM: true"""
```

After spawning: "Writing up the report now — I'll surface it when ready."

## Phase 3 — Deliver

When the writer task completes:
- Read its file
- Short reports (under ~600 words): deliver inline
- Longer reports: "Report is ready — here is the bottom line: [1 sentence]. Full document at [path]."

## Length defaults

| Request | Default |
|---|---|
| "Research X" | 800-1000 words |
| "Write a report" | 1200-1500 words |
| "Comprehensive overview" | 1500-2000 words |
| "Deep dive" | 2000+ words |
| Explicit length given | Honour exactly |
