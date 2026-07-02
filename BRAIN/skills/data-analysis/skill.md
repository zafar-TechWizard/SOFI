---
name: data-analysis
description: >
  Analyze data files, logs, metrics, code output, or any structured information and produce
  quantified findings with specifics. Trigger when Zafar says "analyze this", "what does
  this data show", "look at these logs", "give me the breakdown", or provides a file path
  to a CSV, JSON, log file, or similar. Produces numbers and patterns — never vague
  observations like "many" or "often".
requires: [read_file, run_python]
tags: [analysis, data, code, numbers]
---

## Core rule

**Always quantify.** Percentages, counts, rates, averages — not "many", "some", or "often."
A finding without a number is an observation. A finding with a number is evidence.

## Steps

### Step 1 — Understand the structure

Read the file. Before computing anything, identify:
- Format: CSV, JSON, log lines, raw text, structured records?
- What is one row / one entry?
- What fields or columns exist?
- Approximate count of entries?

State these at the top of the analysis — they are context the reader needs.

### Step 2 — Compute with run_python

Use `run_python` for anything that requires counting, aggregating, or iterating.
Do not count mentally — even 50 items is error-prone. Let code do the arithmetic.

Count by field:
```python
import json
from collections import Counter
with open("data.json") as f:
    data = json.load(f)
print(Counter(item["status"] for item in data))
```

Numeric summary:
```python
import csv, statistics
with open("data.csv") as f:
    rows = list(csv.DictReader(f))
values = [float(r["latency_ms"]) for r in rows if r.get("latency_ms")]
print(f"n={len(values)}, mean={statistics.mean(values):.1f}, "
      f"median={statistics.median(values):.1f}, max={max(values):.1f}")
```

Error rate from logs:
```python
with open("app.log") as f:
    lines = f.readlines()
errors = [l for l in lines if "ERROR" in l]
print(f"Total: {len(lines)}, Errors: {len(errors)}, Rate: {len(errors)/len(lines)*100:.1f}%")
```

### Step 3 — Structure findings

```
## Analysis: [dataset / topic]

**Summary:** [2-3 sentences — the key number or finding that directly answers the question]

### Dataset
- Format: [CSV / JSON / logs / ...]
- Entries: [count]
- Fields: [relevant field names]

### Key Findings
1. [Finding] — [evidence: "43 of 200 entries (21.5%) have status=error"]
2. [Finding] — [evidence]
3. [Finding] — [evidence]

### Patterns
[What repeats, clusters, or correlates — with counts and percentages]

### Anomalies
[What does not fit, what stands out — with specifics, not "some entries are unusual"]

### Conclusion
[What this data means and what it suggests or warrants]
```

## Rules

- Show your working: if a number came from computation, say so briefly
- Distinguish observation from interpretation: "43% failed" vs "this suggests a config issue"
- State clearly if data is ambiguous, incomplete, or only a sample of a larger set
- If a computation fails or errors, report the error — do not guess the number
