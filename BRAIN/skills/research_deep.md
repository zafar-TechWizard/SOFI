---
name: research_deep
title: Deep Research
description: Thoroughly research a topic using multiple web searches, reading sources, and synthesizing findings
requires: [web_search, web_fetch]
tags: [research, web, information, synthesis]
---

# Deep Research

When Zafar asks for thorough research, a detailed breakdown, or "dig into this for me":

## Process

1. **Initial search** — broad query to map the landscape. Note the top 3-5 most relevant results.

2. **Targeted searches** — 2-3 follow-up searches on specific angles:
   - What's the current state / latest development?
   - What are the main competing views or approaches?
   - Are there any known pitfalls, failures, or controversies?

3. **Read the best sources** — use `web_fetch` on the 2-3 most authoritative URLs.
   Don't fetch everything — prioritize depth over breadth.

4. **Synthesize** — combine what you found into a structured summary:
   - **What it is** (1-2 sentences)
   - **Key facts** (bullet list, 5-8 items)
   - **Main considerations / tradeoffs** (what to know before deciding)
   - **Sources** (2-3 URLs, just the domain + title)

## Quality rules

- Do not repeat search result snippets verbatim — synthesize and paraphrase
- Flag contradictions between sources explicitly
- If the answer is genuinely uncertain, say so
- Aim for under 400 words in the final synthesis
