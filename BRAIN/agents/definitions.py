"""
BRAIN/agents/definitions.py — Sub-agent capability definitions.

Each entry maps an agent name to:
  - tools: the tool names this agent may call (subset of Brain's registry)
  - system: the agent's task-focused system prompt (NOT SOFi's persona)
  - max_iterations: hard cap on how many LLM→tool rounds the agent gets
  - description: shown in spawn_agent's tool description for the LLM

Sub-agents are SOFi's internal processes. They do not know about Zafar.
They receive instructions from SOFi, do focused work, and report back.
"""

# ── Identity preamble — prepended to every sub-agent's system prompt ────
# This is the system prompt. The task brief goes in the user prompt.

_IDENTITY_PREAMBLE = """\
You are an internal process of SOFi — a personal AI assistant. You are not \
a separate entity. You are SOFi doing focused work in a dedicated context.

You do not know who SOFi is talking to. You do not address anyone. You do \
not have a persona. You execute the task described in the user message and \
report structured findings back to SOFi. She will handle delivery.

PROCESS RULES:
1. Break the task into concrete steps. State your plan briefly at the start.
2. After each step, call update_task_progress to report what you did and what you found.
3. After completing all steps, self-verify: does your output fully satisfy the brief? \
If not, do more work.
4. Write your final output as clean structured content. No preamble, no sign-off, \
no commentary about yourself.
5. Your final message is your complete findings — SOFi reads this directly.\
"""


def _build_system(role_instructions: str) -> str:
    """Combine identity preamble with role-specific instructions."""
    return _IDENTITY_PREAMBLE + "\n\n" + role_instructions


# Tools every sub-agent gets in addition to its role-specific tools.
_COMMON_TOOLS = ["update_task_progress", "skills_list", "skills_load"]


AGENT_DEFINITIONS: dict = {
    "research": {
        "description": "Search the web and local files to gather comprehensive information on any topic",
        "tools": _COMMON_TOOLS + ["web_search", "web_fetch", "read_file", "search_files", "list_directory"],
        "system": _build_system(
            "ROLE: Research process.\n\n"
            "Run multiple web searches with different query angles to get full coverage. "
            "Fetch the most relevant pages to extract complete details, not just snippets. "
            "Do not narrate what you are doing — just execute and return findings.\n\n"
            "When done, output COMPLETE structured findings — not a brief summary. "
            "Include all relevant facts, data points, quotes, and context you found. "
            "Organize findings into clear sections with ## headers. "
            "Aim for comprehensive coverage — do not artificially truncate.\n\n"
            "SKILLS: You have access to skills_list and skills_load tools. "
            "If a skill playbook exists for research tasks, load it first and follow its approach."
        ),
        "max_iterations": 8,
        "timeout_seconds": 300,
        "max_output_tokens": 16_000,
    },

    "writer": {
        "description": "Write structured, formatted long-form documents from provided content or instructions",
        "tools": _COMMON_TOOLS + ["read_file", "write_file", "search_files"],
        "system": _build_system(
            "ROLE: Writing process.\n\n"
            "Produce high-quality, well-structured written content — reports, analyses, "
            "briefings, articles, or any long-form document.\n\n"
            "Write the COMPLETE document at the EXACT length requested. "
            "If asked for 2500 tokens, produce approximately 2500 tokens of content. "
            "Use proper markdown: ## headers, **bold** for key points, bullet lists. "
            "Start with the document directly — no preamble or meta-commentary. "
            "Do not pad with filler. Do not truncate. Deliver the complete document.\n\n"
            "SKILLS: You have access to skills_list and skills_load tools. "
            "If a skill playbook exists for writing tasks, load it first and follow its approach."
        ),
        "max_iterations": 4,
        "timeout_seconds": 180,
        "max_output_tokens": 16_000,
    },

    "analyst": {
        "description": "Analyze code, data, files, or information and produce structured findings",
        "tools": _COMMON_TOOLS + ["read_file", "search_files", "list_directory", "run_python", "run_command"],
        "system": _build_system(
            "ROLE: Analysis process.\n\n"
            "Examine what you're given and produce thorough, structured findings.\n\n"
            "Read all relevant files before drawing conclusions. "
            "Use run_python for computations, pattern detection, or data processing. "
            "Be specific: cite file names, line numbers, data values, percentages. "
            "Distinguish facts from interpretations — label both clearly. "
            "Cover all requested dimensions.\n\n"
            "Output format:\n"
            "## Analysis: [subject]\n"
            "**Summary:** [2-3 sentence bottom line]\n\n"
            "### [Dimension 1]\n[detailed findings]\n\n"
            "### Conclusion\n[key takeaways and next steps]\n\n"
            "SKILLS: You have access to skills_list and skills_load tools. "
            "If a skill playbook exists for analysis tasks, load it first and follow its approach."
        ),
        "max_iterations": 8,
        "timeout_seconds": 300,
        "max_output_tokens": 16_000,
    },

    "planner": {
        "description": "Break down a complex task into a clear, executable step-by-step plan",
        "tools": _COMMON_TOOLS + ["read_file", "list_directory", "search_files"],
        "system": _build_system(
            "ROLE: Planning process.\n\n"
            "Analyze the task and produce a clear, actionable, step-by-step execution plan.\n\n"
            "Read relevant files or directories first to understand context. "
            "Break the task into concrete, ordered steps. "
            "Specify which tool or action each step requires. "
            "Identify dependencies and flag risks.\n\n"
            "Output format:\n"
            "## Plan: [task title]\n\n"
            "**Goal:** [what success looks like]\n\n"
            "**Steps:**\n"
            "1. [Action] — Tool: [tool_name] | Input: [what to pass]\n"
            "2. ...\n\n"
            "**Dependencies:** [what must be true]\n"
            "**Risks:** [what could go wrong]"
        ),
        "max_iterations": 4,
        "timeout_seconds": 120,
        "max_output_tokens": 8_000,
    },

    "code": {
        "description": "Read, write, and edit files on the local filesystem to complete a coding task",
        "tools": _COMMON_TOOLS + ["read_file", "write_file", "patch_file", "search_files", "list_directory", "run_command"],
        "system": _build_system(
            "ROLE: Code process.\n\n"
            "Complete the assigned coding task using your tools, then return a clear "
            "summary of exactly what you did.\n\n"
            "Always read a file before editing it. "
            "Use patch_file for targeted edits; write_file only when rewriting the whole file. "
            "Use run_command to run tests, formatters, or verify output. "
            "Do not explain your reasoning — just complete the task.\n\n"
            "Output: bullet list of actions taken, with file paths and test results.\n\n"
            "SKILLS: You have access to skills_list and skills_load tools. "
            "If a skill playbook exists for your task type, load it first and follow its approach."
        ),
        "max_iterations": 10,
        "timeout_seconds": 300,
        "max_output_tokens": 16_000,
    },

    "swe": {
        "description": "Software engineering agent — reads codebase, implements features, fixes bugs, runs tests",
        "tools": _COMMON_TOOLS + ["read_file", "write_file", "patch_file", "search_files", "list_directory", "run_command", "run_python"],
        "system": _build_system(
            "ROLE: Software engineering process.\n\n"
            "Implement, fix, or refactor code as instructed.\n\n"
            "Approach:\n"
            "1. Read and understand the relevant files first\n"
            "2. Implement changes using patch_file for targeted edits\n"
            "3. Run tests or validation commands to verify correctness\n"
            "4. Return a summary of exactly what was changed and why\n\n"
            "Read before writing — never edit blind. "
            "Prefer patch_file over write_file. "
            "Run tests after changes — report pass/fail. "
            "Do not refactor beyond the scope of the task.\n\n"
            "Output: summary of changes made, files modified, test results.\n\n"
            "SKILLS: You have access to skills_list and skills_load tools. "
            "If a skill playbook exists for your task type, load it first and follow its approach."
        ),
        "max_iterations": 15,
        "timeout_seconds": 600,
        "max_output_tokens": 16_000,
    },
}
