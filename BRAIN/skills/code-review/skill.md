---
name: code-review
description: >
  Systematic code review across correctness, security, performance, and design. Trigger when
  Zafar says "review this code", "check this file", "look at this for issues", "audit",
  "what is wrong with", or sends a file path expecting quality feedback. For inline review of
  1-3 files. For large codebases (4+ files or full-feature audits), spawn an analyst or swe
  agent and delegate the review.
requires: [read_file, list_directory]
tags: [code, review, analysis, quality]
---

## Before you start

If specific files are named, read them directly.
If a directory or feature is named, run `list_directory` first to understand the structure,
then read the most central files.
Read imports and referenced modules — bugs often live in what is being called, not what is
calling it.

## Four dimensions — review in this order

### 1. Correctness — does it do what it should?

- Logic bugs, wrong conditions, off-by-one errors
- Unhandled edge cases: empty inputs, None/null, boundary values, missing keys
- Wrong assumptions about data types or formats
- Race conditions or async safety issues (especially: awaiting inside sync context, shared mutable state)

### 2. Security — what could be exploited?

- Unvalidated inputs at system boundaries (user input, API responses, file contents)
- Injection risks: SQL, shell commands, OS commands
- Hardcoded secrets, API keys, passwords in source
- Path traversal vulnerabilities (unsanitised file paths from user input)
- Missing auth checks on sensitive operations

### 3. Performance — what breaks under load?

- N+1 query patterns (query inside a loop over results)
- Blocking I/O in async contexts (synchronous network/disk call in an async function)
- Repeated computation inside loops that could be hoisted or cached
- Memory leaks or unbounded accumulation (appending to a list/dict with no eviction)

### 4. Design — is it clear and maintainable?

- Names that mislead (a function that does more than its name says)
- Functions doing too many things (more than ~30 lines with mixed concerns is a signal, not a rule)
- Repeated logic that warrants abstraction (only flag if it actually repeats 3+ times)
- Dead code that is never called
- Missing error handling at external boundaries: API calls, file I/O, subprocess

## Output format

```
## Code Review: [file(s) reviewed]

### Critical
**[Issue title]** — `filename.py:line`
[What the bug is, why it matters, how it can fail in practice]
Fix: [specific code or concrete approach]

### Warnings
**[Issue title]** — `filename.py:line`
[What is wrong]
Fix: [suggestion]

### Minor
- `filename.py:line` — [quick note]

### Verdict
[2-3 sentences: overall quality, the most important fix, what is done well, production-ready yes/no]
```

## Rules

- Cite `file:line` for every finding — a finding without a location is not actionable
- Only flag real issues — style preferences are not bugs unless a style guide was specified
- If no issues found in a dimension, omit that section entirely rather than writing "none found"
- Verdict must state whether the code is production-ready (yes / no / yes after specific fixes)
- For large codebases: say "Spawning analyst agent for a full review" and use spawn_agent
