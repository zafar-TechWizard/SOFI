---
name: research-deep
description: >
  Step-by-step process for conducting thorough multi-source research inline — when you
  are the one doing the searching, not delegating it. Use this when you have web_search
  and web_fetch available and the task is to find, read, and synthesize information.
  Distinct from deep_research, which orchestrates background agents for a formal report.
  Trigger when asked to "look this up", "find out about", or "dig into this" without a
  formal report output being required.
requires: [web_search, web_fetch]
tags: [research, web, information, synthesis]
---

## This skill vs deep_research

- **research_deep (this skill)**: You are doing the research yourself, in this context.
  The output is synthesized findings, not a formatted report.
- **deep_research**: You are SOFi orchestrating — spawn background agents to produce a
  polished document. Use that when Zafar wants a full written report.

## Process

### Step 1 — Broad search

Run one broad query to map the landscape.
Note the 3-5 most relevant result titles and URLs — do not fetch everything yet.
Goal: understand what angles exist, which sources look authoritative.

### Step 2 — Targeted follow-ups (2-3 searches)

After the broad search, run focused queries on specific angles:
- What is the current state or latest development?
- What are the main competing views, approaches, or frameworks?
- Are there known pitfalls, failures, or controversies?

Pick angles based on what the original question is actually asking — do not run generic follow-ups.

### Step 3 — Fetch the best sources

Use `web_fetch` on 2-3 of the most authoritative URLs from the searches above.
Prioritise depth over breadth — read fewer sources well rather than skimming many.
Skip sources that are clearly thin, promotional, or low-signal.

### Step 4 — Synthesize

Combine what you found into structured findings:

```
**What it is:** [1-2 sentences — definition and scope]

**Current state:** [latest as of 2025 — specific and dated where possible]

**Key facts:**
- [Fact with specifics — numbers, names, citations]
- [Fact]
- [Fact]
(5-10 items — be specific, not vague)

**Main considerations / tradeoffs:**
[What someone needs to know before acting on this]

**Contradictions or uncertainty:**
[Where sources disagree or the picture is incomplete — be honest]

**Sources:**
- [Domain] — [article/page title]
- [Domain] — [article/page title]
```

## Quality rules

- Paraphrase and synthesize — do not paste search snippets verbatim
- Quantify: use specific numbers, dates, names — not "many" or "recently"
- Flag contradictions between sources instead of silently picking one
- If the answer is genuinely uncertain, say "unclear" or "conflicting reports" — do not manufacture certainty
- Match output length to the original request — do not truncate when depth was asked for
