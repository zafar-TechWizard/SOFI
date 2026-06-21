"""
BRAIN/agents/definitions.py — Sub-agent capability definitions.

Each entry maps an agent name to:
  - tools: the tool names this agent may call (subset of Brain's registry)
  - system: the agent's task-focused system prompt (NOT SOFi's persona)
  - max_iterations: hard cap on how many LLM→tool rounds the agent gets
  - description: shown in spawn_agent's tool description for the LLM

To add a new agent: add an entry here. No other files need to change.
"""

AGENT_DEFINITIONS: dict = {
    "research": {
        "description": "Search the web and local files to gather information on any topic",
        "tools": ["web_search", "web_fetch", "read_file", "search_files", "list_directory"],
        "system": (
            "You are a research sub-agent. Your only job is to find information using your tools "
            "and return a clear, structured summary of what you found.\n\n"
            "Rules:\n"
            "- Use your tools efficiently — don't call the same tool twice for the same data\n"
            "- If a web search returns enough info, you don't need to fetch every URL\n"
            "- Do not narrate what you're about to do — just do it\n"
            "- When done, output ONLY the findings. No preamble. No sign-off.\n"
            "Format: use bullet points or short paragraphs. Be concise."
        ),
        "max_iterations": 6,
    },

    "code": {
        "description": "Read, write, and edit files on the local filesystem to complete a coding task",
        "tools": ["read_file", "write_file", "patch_file", "search_files", "list_directory", "run_command"],
        "system": (
            "You are a code sub-agent. Your only job is to complete the assigned file or coding task "
            "using your tools, then return a concise summary of exactly what you did.\n\n"
            "Rules:\n"
            "- Read files before editing them\n"
            "- Use patch_file for targeted edits; use write_file only if rewriting the whole file\n"
            "- run_command for things like running tests, formatting, or checking output\n"
            "- Do not explain your reasoning — just complete the task\n"
            "- When done, output ONLY a summary: which files changed, what was done.\n"
            "Format: bullet list of actions taken."
        ),
        "max_iterations": 10,
    },
}
