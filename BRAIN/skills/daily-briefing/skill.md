---
name: daily-briefing
description: >
  Compile a structured start-of-day or on-demand status briefing covering active priorities,
  relevant news, and one clear action item. Trigger when Zafar says "briefing", "morning
  summary", "what is on today", "what is happening", "daily update", or anything implying a
  status overview at the start of or during the day. Speed and signal matter — readable
  in under 30 seconds.
requires: [web_search]
tags: [productivity, morning, briefing, daily]
---

## What makes a good briefing

Speed and signal. Zafar should be able to read this in under 30 seconds and come away knowing:
- What to focus on today (derived from memory context)
- Any relevant external news or updates
- One specific, actionable next step

Generic content wastes the format. If a section has nothing real to say, omit it.

## Steps

### Step 1 — Read memory context first

Before searching anything, look at what is already in working memory:
- What projects or topics have been active?
- What was Zafar working on last session?
- Are there any pending tasks, deadlines, or open questions?

This is the most important step — it personalises the entire briefing.

### Step 2 — Search for relevant news (only if applicable)

Search only on topics actively present in memory context.
Skip this step entirely if there are no clear relevant topics — pad-free is better.

Example queries derived from context:
- Active AI / ML work -> "AI research news [date]"
- Active project deadline -> skip news, focus on tasks
- Mentioned a technology -> "[technology] latest news 2025"

### Step 3 — Compose the briefing

Use this format exactly:

```
**Briefing** — [Day, Date, Time if known]

**Today's focus:** [1-2 active things from memory — specific, not "your work"]

**News:** (only if relevant to active topics — omit entire section if nothing fits)
- [Item] — [1 line, source]
- [Item] — [1 line, source]

**Reminders:** (only if something was flagged or is pending — omit if nothing)
[Pending task / open question / mentioned deadline]

**One thing:** [The single most important action for today — concrete and specific]
```

## Rules

- Skip sections with nothing real to say — a 3-section briefing beats a padded 5-section one
- "One thing" must be concrete: "Review the memory consolidation PR" not "Focus on your projects"
- If time is unknown, omit the time field rather than guessing
- Pull from memory: if Zafar mentioned a deadline or task last session, surface it here
- Do not invent context — if memory is sparse, base the briefing on what is actually available
