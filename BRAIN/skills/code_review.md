---
name: code_review
title: Code Review
description: Review a file or diff for bugs, style, logic issues, and improvements
requires: [read_file, search_files]
tags: [code, review, quality, debugging]
---

# Code Review

When Zafar asks to review code, review a file, or look at a diff:

## Steps

1. **Read the file** — use `read_file` to get the full contents.
   If only a section was described, use `offset_line` to read the relevant part.

2. **Scan for issues** — evaluate on these axes:
   - **Bugs**: logic errors, off-by-ones, null/None handling, uncaught exceptions
   - **Security**: injection risks, hard-coded secrets, unsafe eval/exec
   - **Clarity**: confusing variable names, missing context, overly complex logic
   - **Performance**: N+1 patterns, unnecessary loops, blocking calls in async code
   - **Dead code**: unused variables, unreachable branches, TODO left in

3. **Check related files** if the issue isn't self-contained — use `search_files` to find
   where this function is called or what it depends on.

## Output format

Group findings by severity:

**Critical** — breaks correctness or security. Must fix.
**Minor** — style, naming, mild logic smell. Fix if touching the file anyway.
**Suggestion** — optional improvements worth considering.

End with a one-line verdict: "Looks clean." or "Needs work on [X]."

Keep it tight. If there's nothing wrong, say so — don't invent issues.
