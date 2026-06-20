# SOFi — Project CLAUDE.md

Read this first. This is the entry point for the entire workspace.

---

## 1. What this project is

**SOFi** is Zafar's personal AI companion — a terminal-based conversational
assistant with persistent long-term memory, a unified persona, and dynamic
emotional/mode awareness. Built by Zafar, for Zafar.

The project is intentionally **not a multi-agent system**, **not a chat app**,
and **not a wrapper around an LLM**. It is a single coherent agent — one
self, one voice, one continuous identity across sessions — whose memory,
state, mode, and persona are wired together as her internal organs, not
as plugged-together services.

**Public surface — the only thing external code imports:**

```python
from BRAIN.brain import Brain

brain = Brain()
await brain.setup()
async for token in brain.process(user_message):
    ...
brain.inspect()         # diagnostic snapshot (state, mode, memory)
brain.force_mode(name)  # manual mode override; 'auto' clears
brain.clear_history()   # wipe Brain-local short history (graph untouched)
await brain.shutdown()
```

That's the whole external API. Everything else lives inside.

---

## 2. The mission constraint

SOFi has to feel like one being having one continuous conversation with one
person across days, months, years. Not like a service that retrieved some
facts. Not like a chatbot wrapping a model. The architecture is in service
of that goal — every design choice trades off against it.

Locked design goals (from previous build phases — these aren't open):

| Goal | Status |
|---|---|
| Single LLM call per turn | Locked — multi-call pipelines kill latency and break voice |
| No LLM in routing / mode / state inference | Locked — those are non-LLM rule-based for determinism + ~1ms cost |
| Streaming response from day 1 | Locked — voice/TTS later, no rewrites |
| Persona is self-truth, not stage direction | Locked — first-person framing throughout |
| Honest about capabilities; never fabricates | Locked — no inventing weather, time, location, memory |
| Jarvis-hybrid: feminine, dry, composed, brief | Locked — full personality spec in `BRAIN/persona/personality.json` |

---

## 3. High-level architecture

```
                     ┌─────────────────────────────────────────┐
                     │                sofi.py                  │
                     │  (terminal entry — loads .env, starts   │
                     │   Brain, drives CLI loop)               │
                     └────────────────┬────────────────────────┘
                                      │
                ┌─────────────────────▼─────────────────────┐
                │            BRAIN/ui/cli.py                │
                │  (Rich + prompt_toolkit; slash commands;  │
                │   live streaming render; status line)     │
                └─────────────────────┬─────────────────────┘
                                      │
                ┌─────────────────────▼─────────────────────┐
                │              BRAIN/brain.py               │
                │  (the coordinator — owns Memory, LLM,     │
                │   StateInferencer, ModeController)        │
                └────┬─────────┬──────┬────────┬───────────-┘
                     │         │      │        │
        ┌────────────▼─┐  ┌────▼───┐ ┌▼──────┐ ┌▼──────────────┐
        │  memory/     │  │ state/ │ │ mode/ │ │ persona/      │
        │ (Neo4j +     │  │ user_  │ │ con-  │ │ + LLM (Groq)  │
        │  retrieval   │  │ state  │ │ trol- │ │ + prompt      │
        │  + working   │  │ infer  │ │ ler   │ │ builder       │
        │  memory)     │  │ -encer │ │       │ │               │
        └──────────────┘  └────────┘ └───────┘ └───────────────┘
```

Three logical layers:

1. **L1 — Working Memory** (`memory/working_memory/`) — in-process RAM, the
   active whiteboard for the current session. Sub-ms reads. Expires per
   session.

2. **L2 — Long-Term Memory** (`memory/long_term/`) — Neo4j graph database,
   persistent across all sessions. Holds `ExperienceMemory`,
   `KnowledgeMemory`, and `RelationshipMemory` nodes connected by 23 typed
   relationship edges.

3. **Processing** (`memory/processing/`) — entity extraction (GLiNER +
   spaCy), embeddings (MiniLM), conversation logging, nightly
   consolidation (Gemini CLI agent).

On top of L1+L2, the **BRAIN/** package adds:

- **Persona** — Jarvis-hybrid SOFi as first-person self-statement
- **State inferencer** — rule-based user-state detection (emotion, need, engagement)
- **Mode controller** — non-LLM multi-signal mode selection with hysteresis
- **Prompt builder** — assembles persona + state + memory into the system prompt
- **LLM client** — async streaming Groq wrapper
- **UI** — Rich-based terminal

---

## 4. Per-turn data flow (the canonical loop)

Every user message follows the same path. Latency annotations are warm-state.

```
user message in terminal
  │
  ▼
[sofi.py reads via prompt_toolkit]
  │
  ▼
[BRAIN/ui/cli.py — input handler]
  │
  ▼
brain.process(message):
  │
  │ 1. await memory.observe("user", message)             ← 0ms (non-blocking)
  │    └─ fires reactive_processing in a background thread
  │       (entity extraction, intent classification, LTM retrieval)
  │
  │ 2. await memory.get_context_async()                  ← 800-1100ms
  │    └─ waits for #1 to finish, returns WorkingContext snapshot
  │
  │ 3. ctx = memory.get_full_context()
  │    └─ four pillars: memory / sofi / user / workspace
  │
  │ 4. user_state = UserStateInferencer.infer(ctx, msg)  ← ~1ms
  │    └─ writes emotion / intensity / need / engagement
  │       into ctx.user via context_manager.update_user_state()
  │
  │ 5. mode_decision = ModeController.decide(ctx, msg)   ← ~0.1ms
  │    └─ returns mode + allow_dropped_formality
  │
  │ 6. system_prompt = build_prompt(ctx, mode, allow_dropped_formality)
  │    └─ persona block + current moment + user state + memory tiers
  │
  │ 7. messages = build_messages(ctx, message)
  │    └─ recent turns from memory + the current message
  │
  │ 8. async for token in GroqClient.stream(system_prompt, messages):
  │       yield token                                    ← 300-2000ms (Groq)
  │
  │ 9. await memory.observe("assistant", response)       ← 0ms (fire-and-forget)
  │
  ▼
[CLI renders tokens live to Rich panel + status line]
```

**Total per turn:** ~1.3-2.5s end-to-end (memory + LLM combined). Most of
the time is the LLM. Brain code itself is < 30ms.

---

## 5. Directory map

```
assistant/                              ← workspace root
├── CLAUDE.md                           ← this file (project-level spec)
├── sofi.py                             ← terminal entry point
├── .env                                ← GROQ_API_KEY + other config
├── requirements.txt
│
├── BRAIN/                              ← the brain subsystem
│   ├── CLAUDE.md                       ← BRAIN-level spec + design log
│   ├── __init__.py
│   ├── brain.py                        ← Brain coordinator class
│   │
│   ├── persona/                        ← identity + voice
│   │   ├── persona.py                  ← get_identity_block(mode, ...)
│   │   └── personality.json            ← Jarvis-hybrid SOFi as self-statement
│   │
│   ├── state/                          ← user/sofi state inferencers
│   │   ├── user_state.py               ← UserStateInferencer (emotion, need, engagement)
│   │   └── self_model.py               ← (PLANNED — capabilities registry)
│   │
│   ├── mode/                           ← mode controller
│   │   ├── controller.py               ← ModeController (4 modes, hysteretic)
│   │   └── signals.py                  ← lexical signal extraction
│   │
│   ├── prompt/                         ← prompt assembly
│   │   ├── builder.py                  ← build_prompt() + build_messages()
│   │   └── formatters.py               ← memory dict → readable line
│   │
│   ├── llm/                            ← LLM client
│   │   └── groq_client.py              ← async streaming wrapper
│   │
│   ├── ui/                             ← terminal interface
│   │   └── cli.py                      ← Rich + prompt_toolkit
│   │
│   ├── memory/                         ← runtime data (NOT source)
│   │   └── data/                       ← conversation.json, working_context.json, logs/, reviews/
│   │
│   └── _test_phase{1,2,3}.py           ← phase acceptance tests
│
├── memory/                             ← the memory subsystem
│   ├── CLAUDE.md                       ← memory-level spec
│   ├── current-gap.md                  ← live state of known issues
│   ├── todo.md                         ← brain build plan reference
│   ├── memory_manager.py               ← PUBLIC INTERFACE
│   ├── config.py                       ← singleton config
│   ├── observability.py                ← observer singleton
│   │
│   ├── long_term/                      ← L2 — Neo4j graph
│   │   ├── memory_retrieval_engine.py
│   │   ├── memory_router.py
│   │   ├── reranker.py
│   │   ├── models/                     ← FROZEN node + relationship models
│   │   └── infrastructure/             ← FROZEN Neo4j + Docker
│   │
│   ├── working_memory/                 ← L1 — RAM whiteboard
│   │   ├── working_mem.py
│   │   ├── working_context.py
│   │   ├── context_manager.py
│   │   └── workspace_watcher.py
│   │
│   └── processing/                     ← entity extraction, consolidation, embedding
│       ├── entity_extractor.py
│       ├── embedding_utils.py
│       ├── consolidation.py
│       ├── consolidation_runner.py
│       ├── conversationLogger.py
│       └── _test_retrieval.py
│
└── (legacy / experimental — not part of V1 SOFi)
    ├── Hand/, TTS/, core/, utils/      ← older multi-agent explorations
    ├── chatbot.py, main.py             ← earlier prototypes
    ├── README.md                       ← describes an older multi-agent vision; superseded by this CLAUDE.md
    └── various test_*.py, *.py         ← scratch / one-off
```

The V1 SOFi system is contained in **`BRAIN/`, `memory/`, and `sofi.py`**.
Everything else at the workspace root is either legacy or unrelated.

---

## 6. Subsystem deep-dives

### 6.1  `memory/` — three-tier memory

**See `memory/CLAUDE.md` for the full spec.** Quick summary:

- **L1 working memory**: per-session RAM, holds active entities,
  tiered memories surfaced this turn, retrieval metadata, emotional
  baseline.
- **L2 Neo4j graph**: cross-session persistent memory. Three node types
  (`ExperienceMemory`, `KnowledgeMemory`, `RelationshipMemory`), 23 typed
  relationship edges (CAUSED, EXPERIENCE_CHAIN, KNOWLEDGE_HIERARCHY, etc.).
  BM25 full-text index + spreading activation traversal.
- **Processing**: entity extraction (GLiNER for named entities + spaCy
  fallback + coreferee), MiniLM-L6-v2 embeddings (used at consolidation
  time only — not on hot path), nightly consolidation via Gemini CLI agent.

**Memory's public API:**

```python
from memory.memory_manager import MemoryManager

manager = MemoryManager(log=False, review=False)
await manager.setup()                              # warms everything

await manager.observe(role, content)               # non-blocking
ctx_dict = await manager.get_context_async()       # async-safe wait
full_context = manager.get_full_context()          # WorkingContext snapshot

await manager.shutdown()
```

### 6.2  `BRAIN/persona/` — Jarvis-hybrid SOFi as self-statement

The persona is written **in first person** in `personality.json` —
`I am SOFi`, not `You are SOFi`. The persona module wraps this with a
preamble: *"What follows is not a description of a character to play. It
is my own inner self — what I know about myself the way anyone silently
knows who they are."*

This shifts the model from playing-a-character mode into being-a-self
mode. Every line in the prompt reads as the model's own self-truth.

Persona has 4 modes (`conversational`, `empathetic`, `focused`, `creative`),
two earned exceptions (name-drop and dropped formality), a quirks list
(including the flirt-handling deflection pattern), and a `never` list with
paired wrong/right templates.

Token cost per turn: ~2,250 tokens (down from ~3,800 after compression on
2026-06-14).

### 6.3  `BRAIN/state/user_state.py` — UserStateInferencer

Rule-based, ~1ms per call. Inputs:
- `intent` from memory's retrieval_meta
- `emotional_baseline` from memory
- message text (regex on emotion keywords, punctuation density, ALL-CAPS)
- prev_state (smoothing/decay)

Outputs written to `WorkingContext.user`:
- `current_emotional_state` ∈ {neutral, stressed, sad, frustrated, overwhelmed, excited, content, focused, tired}
- `emotional_intensity` ∈ [0, 1]
- `current_need` ∈ {emotional_support, practical, informational, creative, casual}
- `engagement_level` ∈ {disengaged, normal, highly_engaged}

Intensity decay: 0.7× per turn when prev was ≥ 0.25 (gives empathetic mode
multi-turn persistence after emotional disclosure).

### 6.4  `BRAIN/mode/` — ModeController

Non-LLM, signal-scored, hysteretic. 4 modes. Score categories:

| Signal | Range | Source |
|---|---|---|
| Intent bias | +0.25 to +0.60 | memory.retrieval_meta.intent |
| User need bias | +0.35 to +0.55 | UserStateInferencer |
| Emotional intensity | +0.20 to +0.40 | UserStateInferencer |
| Message lexical signals | +0.30 to +0.50 | regex (`BRAIN/mode/signals.py`) |
| Message shape (length, punctuation) | +0.15 to +0.30 | message stats |
| Hysteresis (prev mode) | +0.20 (or +1.00 after hard override) | tracked across turns |

**Hard overrides** (skip scoring, return immediately):
1. `emotional_intensity ≥ 0.7` → empathetic
2. Code block present (``` `) → focused
3. Explicit creative phrase ("brainstorm with me", "help me design") → creative

**Stability**:
- Margin gate: winner must beat second by ≥ 0.15
- Default: conversational on all-zero scores
- After a hard override, the winning mode gets +1.00 hysteresis next turn

Outputs: `(Mode, allow_dropped_formality: bool)`. The dropped-formality
flag fires when `mode == empathetic and intensity ≥ 0.6` — it unlocks the
persona's earned exception (one sentence may drop "sir").

### 6.5  `BRAIN/prompt/` — Prompt assembly

`build_prompt(ctx, mode, allow_dropped_formality)` assembles:

```
[persona preamble — self-truth framing]
━━━ Who I am ━━━              (identity)
━━━ What is real about me ━━━ (current_truth — capabilities + limits)
━━━ My character ━━━
━━━ My worldview ━━━
━━━ How I speak ━━━
━━━ How I address Zafar ━━━
━━━ My quirks ━━━             (incl. flirt-handling pattern)
━━━ What I never do ━━━       (paired wrong/right examples)
━━━ How I am right now ━━━    (mode-specific behaviour)
━━━ Speak ━━━                 (mode-specific closing instruction)

━━━ CURRENT MOMENT ━━━        (date/time/time-of-day from SofiState)
━━━ WHAT'S TRUE FOR ZAFAR RIGHT NOW ━━━   (mentioned entities, current focus)
━━━ WHAT YOU REMEMBER (most relevant) ━━━ (must_know memories — formatted)
━━━ BACKGROUND CONTEXT ━━━                (context tier)
━━━ LOOSELY RELATED ━━━                   (associations tier)
```

`build_messages(ctx, message)` returns the Groq `messages` list using
working memory's `recent_turns`, with a defensive fallback to Brain-local
history if memory's list is empty.

`BRAIN/prompt/formatters.py` turns raw memory dicts into compact
single-line bullets with optional age + emotion qualifiers.

### 6.6  `BRAIN/llm/groq_client.py` — Groq streaming client

Thin async wrapper around the `groq` SDK. One method:

```python
async def stream(system_prompt: str, messages: List[Dict[str, str]]) -> AsyncIterator[str]
```

Default model: `llama-3.3-70b-versatile`. Temperature 0.7. Max tokens 1024.
API key from `GROQ_API_KEY` env var. Surfaces errors honestly — does not
swallow failures into "sorry, error".

### 6.7  `BRAIN/ui/cli.py` — Terminal UI

Stack: `rich` (output, live streaming, panels, status line) +
`prompt_toolkit` (input with history, multi-line). Python equivalent of
the Ink/React stack used by Gemini CLI and Claude Code.

**Slash commands**:
- `/exit`, `/quit`, `/q` — quit
- `/clear` — wipe in-session conversation (memory graph untouched)
- `/mode <name>` — force mode (conversational | empathetic | focused | creative)
- `/mode auto` — return to controller-driven
- `/status` — full state snapshot
- `/memory` — memories surfaced last turn
- `/help` — help panel

**Cold-start greeting**: short Jarvis-coded line based on time of day
("Good morning, sir." / "Afternoon, sir." / "Evening, sir." / "Late one, sir.").

**Status line** under each response: `mode=X · emotion=Y · intensity=Z ·
first-token Xms · total Yms`. Computed *after* the stream finishes so it
reflects the current turn's state (was a stale-one-turn bug, fixed in
Phase 4).

---

## 7. How to run

### 7.1  Prerequisites

- Python 3.11
- Docker Desktop (for Neo4j container)
- A Groq API key (free tier works; paid Dev Tier recommended for daily use)

### 7.2  One-time setup

```bash
# Install deps
pip install -r requirements.txt

# Set up .env at workspace root
echo 'GROQ_API_KEY=gsk_...your_key_here...' >> .env
```

Neo4j is auto-managed via Docker by `memory/long_term/infrastructure/docker_manager.py` — no manual container setup.

### 7.3  Start SOFi

```bash
python sofi.py
```

First boot is ~30-60 seconds (Docker + GLiNER + cross-encoder + end-to-end pipeline warmup). Subsequent boots within the same Docker session are faster.

After boot, you see the cold-start greeting and the input prompt. Type and converse. `/exit` to quit cleanly.

### 7.4  Run consolidation (manually)

Consolidation transforms recent conversation logs into graph memories.
Currently manual; future work: schedule it.

```bash
python -m memory.processing.consolidation_runner
```

### 7.5  Run the test suites

```bash
# Memory L1+L2+L3 tests
python -X utf8 -m memory.processing._test_retrieval

# Brain Phase 1 (persona-only)
python -X utf8 -m BRAIN._test_phase1

# Brain Phase 2 (persona + memory)
python -X utf8 -m BRAIN._test_phase2

# Brain Phase 3 (full: persona + memory + state + mode)
python -X utf8 -m BRAIN._test_phase3
```

---

## 8. Configuration

### 8.1  Environment variables (`.env`)

| Var | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq API key for the LLM client |

### 8.2  Tunable settings (`memory/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer for consolidation embeddings |
| `entity_expiry_minutes` | 15 | Active entities expire from working memory after this gap |
| `context_retrieval_timeout_ms` | 1500 | Hard cap on `get_context_async()` wait |
| `working_context_recent_turns` | 5 | How many recent turns to surface to the prompt |
| `session_timeout_minutes` | 30 | ConversationLogger starts a new session after this gap |
| `neo4j_uri` | `bolt://localhost:7687` | Neo4j connection |
| `neo4j_password` | `SofiAiAssistant` | Default password baked into the Docker container |

---

## 9. Component status

### Fully built + tested

| Component | Status | Notes |
|---|---|---|
| MemoryManager (public interface) | ✅ | L1+L2+L3 tests pass |
| Three-tier memory (RAM + Neo4j + processing) | ✅ | Full data flow validated |
| Intent classifier | ✅ | 5 intents, regex, ~0ms |
| AMBIENT bypass | ✅ | Greetings skip LTM |
| RF-Mem familiarity gate | ✅ | Skips LTM when entities are warm |
| Entity extraction (GLiNER + spaCy + sliding window) | ✅ | ~5-20ms warm |
| Active entity propagation | ✅ | |
| BM25 full-text + spreading activation | ✅ | Typed-edge traversal |
| ACT-R heat scoring | ✅ | |
| Cross-encoder reranker | ✅ | ms-marco-MiniLM-L-6-v2, 22MB |
| Coverage check + backup queries | ✅ | |
| Hebbian reinforcement | ✅ | Fire-and-forget |
| Memory router (full pipeline) | ✅ | All 5 intents route correctly |
| Working memory L1 | ✅ | Thread-safe state, expiry, sliding window |
| `recent_turns` updates on every observe | ✅ | (Was bug, fixed) |
| WorkingContext (four pillars) | ✅ | |
| AgenticWorkspace (scaffolding for proactivity) | ✅ | Not yet wired to brain |
| WorkspaceWatcher daemon | ✅ | Updates SofiState time fields; proactive callback stub |
| Conversation logger (with sessions) | ✅ | UUID per session, 30-min timeout |
| Consolidation pipeline (Gemini CLI) | ✅ | Runs manually; 7/7 sessions validated |
| `get_context_async` (async-safe wait) | ✅ | Sync version freezes the event loop |
| End-to-end pipeline warmup at setup | ✅ | First user message lands warm |
| BRAIN/persona (Jarvis-hybrid SOFi) | ✅ | First-person self-statement, compressed to ~2250 tokens |
| BRAIN/state/user_state.py | ✅ | Rule-based ~1ms |
| BRAIN/mode/controller.py (4 modes) | ✅ | Multi-signal + hysteresis + hard overrides + margin gate |
| BRAIN/mode/signals.py | ✅ | 8 lexical signal categories |
| BRAIN/prompt/builder.py + formatters.py | ✅ | |
| BRAIN/llm/groq_client.py | ✅ | Async streaming |
| BRAIN/ui/cli.py | ✅ | Rich + prompt_toolkit; 6 slash commands |
| BRAIN/brain.py (coordinator) | ✅ | Owns memory, LLM, state, mode |
| sofi.py (entry point) | ✅ | |

### Stubbed / Brain-fills-them

| Component | Status | Notes |
|---|---|---|
| SofiState.emotional_tone / energy_level / current_focus | ⚠️ Brain fills via UserStateInferencer + Mode | Working but minimal |
| AgenticWorkspace → brain (proactivity wire) | ⚠️ Stub | WorkspaceWatcher fires `_on_proactive_notification` but it's a no-op |

### Planned (next)

| Component | Priority | Effort |
|---|---|---|
| BRAIN/state/self_model.py (capability registry) | **Next** | ~1.5h |
| Better memory formatters | Tier 1 polish | 30 min |
| Consolidation scheduling (auto-run on shutdown or cron) | Tier 1 polish | 1h |
| Tool calling V1 (one tool to prove the pattern) | Tier 2 | 1 day |
| Proactivity (WorkspaceWatcher → brain.speak_unprompted) | Tier 2 | half day |

### Deferred / Future

| Component | When |
|---|---|
| STT / TTS streaming pipeline | After V1 use |
| Skip cross-encoder when results well-separated | When latency complaints arise |
| Edge-type pruning for spreading activation | Once graph has 500+ memories |
| Consolidation smoke test | If consolidation breaks while iterating |
| Startup health validation | Deployment polish |
| MMR diversity ranking | When usage shows repetitive memories |
| ML intent classifier | When regex fails |
| Concept graph | When multi-topic failures show up |
| Cross-encoder fine-tuning | When months of real conversation data exist |

---

## 10. Known issues / current gaps

Tracked in `memory/current-gap.md` (authoritative for memory-side issues).
At project level:

1. **Groq free tier throttles daily use.** 100k tokens/day = ~29 turns at
   current persona size (~3,460 tokens/turn after the 2026-06-14
   compression). Path forward: upgrade to Groq Dev Tier (~$0.003/turn) for
   actual daily use. No code change needed.

2. **Sir frequency calibration is iterative.** Has swung from too-sparse
   (1 in 19) to too-frequent (nearly every turn). Current rule is a hard
   ceiling of 1 in 3. Persona-level fix; may need further tuning.

3. **`emotional_baseline` only populates for EMOTIONAL intent.** Means
   non-emotional turns don't get memory-side affect signal. Works as
   designed but worth noting.

4. **Consolidation is manual.** Must be triggered via
   `python -m memory.processing.consolidation_runner`. Until scheduling
   is built, memory only grows when you remember to run it.

5. **No multilingual emotion detection.** Hindi/Hinglish emotional
   keywords don't fire the user_state inferencer. By design — relying on
   memory baseline + punctuation + intent for cross-language emotional
   sensing. Acceptable for V1.

---

## 11. Build history (phases completed)

| Phase | What was built | Acceptance |
|---|---|---|
| **Memory L1+L2+L3** | Full memory subsystem with consolidation | Three-level test passes |
| **BRAIN Phase 1** | Persona + Groq + CLI | `_test_phase1.py` passes |
| **BRAIN Phase 2** | + Memory wiring (observe/get_context_async) | `_test_phase2.py` passes |
| **BRAIN Phase 3** | + UserStateInferencer + ModeController | `_test_phase3.py` passes (7/7) |
| **BRAIN Phase 4** | + Slash commands, status line, inspector | Manual test |
| **Persona refinement #1** | Behavioral never list, address rules, earned exceptions | Manual test |
| **Persona refinement #2** | Embedded feminine identity, digital-human framing | Manual test |
| **Persona compression + first-person self-truth framing** | Tokens 3800 → 2250 per turn | Smoke verified 2026-06-14 |
| **Self-model module** | (PLANNED — next) | TBD |
| **Tool calling V1** | (PLANNED — Tier 2) | TBD |
| **Proactivity wire** | (PLANNED — Tier 2) | TBD |

---

## 12. Key design decisions (the "why" log)

| Decision | Reasoning |
|---|---|
| **Single LLM call per turn** | Multi-call pipelines add 100-300ms each and break voice consistency. The model already does intent + emotion + response in one shot. |
| **No LLM in routing / mode / state** | Determinism, sub-1ms cost, debuggability. Regex+scoring covers 95% of cases. |
| **Persona as first-person self-statement** | Empirically, models internalize "I am X" more strongly than "You are X". Shifts the model from playing a character to being one. |
| **Hard ceiling on `sir` frequency (1 in 3)** | Soft "natural" guidance produced verbal-tic over-use. Hard ceiling + no-address default works. |
| **`recent_turns` updated on every observe (not just non-AMBIENT)** | Was a bug — AMBIENT turns left WorkingContext.memory.recent_turns stale. |
| **`get_context_async` for async callers** | Sync version's `threading.Event.wait()` froze the event loop, starving bridged Neo4j calls. |
| **End-to-end pipeline warmup at MemoryManager.setup()** | First user message would otherwise pay 800-1500ms of cold-start cost. |
| **Memory observes pass through one logger; conversation_logger.log_message is sync** | Was async fire-and-forget, raced with the recent_turns read. Sync is fine — disk write is fast. |
| **FACTUAL with no entities falls back to BM25 over message keywords ≥ 5 chars** | Without this, knowledge queries with no extracted entities returned nothing. No stop-word list — Lucene scores common short words near zero anyway. |
| **`current_truth.cannot_do` is part of every turn's prompt** | Stops hallucination of weather/time/location. SOFi sees her own limits and acknowledges them in her own voice. |
| **Memory uses Brain-local short history as defensive fallback** | If `WorkingContext.recent_turns` somehow returns empty, Brain's own list keeps the conversation coherent. |
| **Streaming from day 1** | Voice/TTS later. No batch-to-stream rewrite needed if it's always streaming. |
| **Sir/Mr. Zafar/Zafar tiered address with earned exceptions** | The Jarvis arc. Rare name-drop signals real emotional weight; dropped formality unlocks empathy without breaking frame. |

---

## 13. Where to read next

| Reader | Start with |
|---|---|
| Brand new to the project | This file (`CLAUDE.md`), then `BRAIN/persona/personality.json` |
| Working on memory | `memory/CLAUDE.md`, then `memory/memory_manager.py`, then `current-gap.md` |
| Working on Brain | `BRAIN/CLAUDE.md`, then `BRAIN/brain.py`, then mode/state/prompt modules |
| Tuning the persona | `BRAIN/persona/personality.json` (data) → `BRAIN/persona/persona.py` (assembly) |
| Debugging a turn | Set `MemoryManager(log=True, review=True)` in Brain; review traces land in `BRAIN/memory/data/reviews/observe/` |

---

## 14. Glossary (terms used throughout)

- **L1 / L2** — short for "Layer 1 / Layer 2" memory: working memory vs Neo4j graph
- **Intent** — 5-class label from memory router (ENTITY, FACTUAL, EMOTIONAL, TEMPORAL, AMBIENT)
- **Mode** — 4-class label from Brain's mode controller (conversational, empathetic, focused, creative)
- **Hard override** — a signal strong enough to skip the mode controller's scoring and pick the mode directly
- **Earned exception** — a persona rule that's normally locked but unlocks under specific conditions (e.g., first-name "Zafar" instead of "sir" in weighty emotional moments)
- **Familiarity gate** — RF-Mem optimization: skip LTM retrieval when all entities are warm in working memory
- **Recent turns** — last N conversation turns surfaced into the prompt as `messages` for Groq
- **Working context** — the four-pillar snapshot returned by `memory.get_full_context()`: `memory`, `sofi`, `user`, `workspace`
- **Self-truth framing** — the persona is written in first person as SOFi's inner self-statement, not as instructions about a character
