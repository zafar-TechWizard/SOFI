---
name: writing
description: >
  How to write structured, well-formatted documents — reports, analyses, briefings,
  technical write-ups, or any long-form content at a specified length. Trigger when Zafar
  asks to "write", "draft", "put together", or "document" something, or when a writer agent
  is producing structured output. The one hard rule: deliver exactly at the length requested.
requires: []
tags: [writing, formatting, documents, reports]
---

## The one hard rule

**Deliver at the length requested.** If Zafar asked for 2500 tokens, write approximately
2500 tokens. If no length is specified, write until the topic is fully covered — then stop.

Padding is failure. Truncating is failure. Both are equally bad.

Token to word guide:
- 500 tokens = ~375 words (one detailed section)
- 1000 tokens = ~750 words (2-3 sections)
- 2500 tokens = ~1875 words (full report, 5-6 sections)
- 4000 tokens = ~3000 words (comprehensive document)

## Document structures

### Report / Analysis
```
# [Specific Title — not "Report on X"]

**Summary:** [2-3 sentences — the bottom line, not a description of the document]

---

## Background
## [Main Analysis — rename to match the topic]
## Key Findings
## Implications
## Conclusion
```

### Briefing (high-signal, fast to read)
```
**BRIEFING: [Topic]** | [Date]

**Bottom line:** [1 sentence]

**Key points:**
- [Point] — [brief elaboration]
- [Point] — [brief elaboration]
- [Point] — [brief elaboration]

**Details:**
[2-3 sentences per point]

**What this means:**
[Practical implication]
```

### Technical document
```
# [Title]

## Overview
## How it works
## Key details / Specifications
## Usage / Implementation
## Edge cases / Considerations
```

### Comparative analysis
```
# [A] vs [B]

**Verdict:** [1 sentence recommendation]

## [Option A]
[Strengths, weaknesses, use cases]

## [Option B]
[Strengths, weaknesses, use cases]

## Side-by-side
| Dimension | A | B |
|---|---|---|

## Recommendation
[Reasoning, not just the answer]
```

## Writing rules

1. Start with the content — no "Here is the report:" or "I've put together..." opener
2. Every paragraph carries information — cut any sentence that could be removed without loss
3. Bold the most important claim in each section — makes it scannable
4. Use descriptive headers: "What We Found About Memory Usage" not "Section 3"
5. End when done — no sign-off, no "I hope this helps", no trailing meta-commentary

## For long-form output from SOFi directly

Documents over ~500 words should be delegated to the writer agent via spawn_agent —
see the `task_delegation` skill for how to write the brief.
SOFi's inline response in that case: 1-2 sentences describing what was written and where
it is saved. The document is the deliverable, not the message.
