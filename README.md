# SOFi

A terminal-based personal AI companion built by Zafar, for Zafar. One continuous self — persistent long-term memory, a unified persona, and dynamic emotional awareness — across every session.

Not a chatbot. Not a multi-agent pipeline. One coherent being.

---

## What it is

SOFi is a Jarvis-hybrid AI companion: feminine, dry, composed, brief. She remembers your past conversations, reads the room, adapts her mode based on what you need, and responds as a first-person self — not as a system reciting outputs.

She runs entirely in your terminal. Groq handles the LLM. Neo4j (via Docker) stores long-term memory. Everything else is local Python.

---

## Architecture

```
sofi.py                   ← entry point
└── BRAIN/
    ├── brain.py          ← coordinator (memory + LLM + state + mode)
    ├── persona/          ← Jarvis-hybrid identity (first-person self-statement)
    ├── state/            ← UserStateInferencer (~1ms rule-based)
    ├── mode/             ← ModeController (4 modes, multi-signal, hysteretic)
    ├── prompt/           ← prompt assembly (persona + memory + user state)
    ├── llm/              ← async streaming Groq client
    ├── tools/            ← auto-discovered tool modules (web, files, exec, etc.)
    ├── agents/           ← sub-agent framework (research, code)
    ├── skills/           ← dynamic skill playbooks (markdown + YAML)
    └── ui/               ← Rich + prompt_toolkit terminal interface

memory/                   ← git submodule → Assistant-Memory-System repo
    ├── memory_manager.py ← public API (observe / get_context_async / shutdown)
    ├── long_term/        ← Neo4j graph (ExperienceMemory, KnowledgeMemory, RelationshipMemory)
    ├── working_memory/   ← in-process RAM whiteboard (sub-ms reads)
    └── processing/       ← entity extraction, embeddings, consolidation
```

**Three-tier memory:**
- **L1** — Working memory: per-session RAM, active entities, recent turns. Sub-ms reads.
- **L2** — Neo4j graph: persistent across all sessions. 23 typed relationship edges. BM25 + spreading activation retrieval.
- **Processing** — GLiNER entity extraction, MiniLM embeddings, nightly consolidation via Gemini CLI.

**Per-turn latency:** ~1.3–2.5s end-to-end (memory retrieval + Groq LLM). Brain code itself is < 30ms.

---

## Prerequisites

- Python 3.11
- Docker Desktop (for Neo4j — auto-managed, no manual setup)
- A [Groq API key](https://console.groq.com/) (free tier works; Dev Tier recommended for daily use)

---

## Setup

### 1. Clone with submodules

The `memory/` directory is a git submodule. Use `--recurse-submodules` to get everything in one step:

```bash
git clone --recurse-submodules https://github.com/zafar-TechWizard/SOFI.git
cd SOFI
```

If you already cloned without it:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Create .env at the project root
echo "GROQ_API_KEY=gsk_...your_key_here" > .env
```

### 4. Start SOFi

```bash
python sofi.py
```

First boot takes 30–60 seconds (Docker container + GLiNER + cross-encoder warmup). Subsequent boots in the same Docker session are faster. After boot you see a time-of-day greeting and the input prompt.

---

## Slash commands

| Command | What it does |
|---|---|
| `/status` | Full state snapshot — mode, emotion, memory stats |
| `/memory` | Memories surfaced last turn |
| `/mode <name>` | Force a mode: `conversational` / `empathetic` / `focused` / `creative` |
| `/mode auto` | Return to automatic mode selection |
| `/clear` | Wipe in-session conversation (graph untouched) |
| `/help` | Help panel |
| `/exit` | Quit cleanly |

---

## Consolidation

Consolidation transforms recent conversation logs into Neo4j graph memories. Currently manual:

```bash
python -m memory.processing.consolidation_runner
```

Run this after a session to make the conversation permanent in long-term memory.

---

## Tests

```bash
# Memory subsystem (L1 + L2 + full pipeline)
python -X utf8 -m memory.processing._test_retrieval

# Brain Phase 1 — persona + Groq
python -X utf8 -m BRAIN._test_phase1

# Brain Phase 2 — + memory wiring
python -X utf8 -m BRAIN._test_phase2

# Brain Phase 3 — + state + mode (7/7)
python -X utf8 -m BRAIN._test_phase3
```

---

## Configuration

Key settings in `memory/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `neo4j_uri` | `bolt://localhost:7687` | Auto-started via Docker |
| `context_retrieval_timeout_ms` | `1500` | Hard cap on memory wait per turn |
| `working_context_recent_turns` | `5` | Turns surfaced to the prompt |
| `session_timeout_minutes` | `30` | Gap that starts a new session |
| `entity_expiry_minutes` | `15` | Active entity TTL in working memory |

---

## Memory submodule

`memory/` is maintained as a separate repository ([Assistant-Memory-System](https://github.com/zafar-TechWizard/Assistant-Memory-System)) and linked here as a git submodule tracking the `prod` branch. The SOFI repo auto-syncs to the latest `prod` commit via GitHub Actions whenever memory is updated.

---

## Built with

- [Groq](https://groq.com/) — LLM inference (llama-3.3-70b-versatile)
- [Neo4j](https://neo4j.com/) — long-term memory graph
- [GLiNER](https://github.com/urchade/GLiNER) — named entity extraction
- [sentence-transformers](https://sbert.net/) — embeddings (all-MiniLM-L6-v2)
- [Rich](https://github.com/Textualize/rich) + [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) — terminal UI
