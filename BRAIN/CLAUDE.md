# SOFi Brain — CLAUDE.md

Read this before touching any file in `BRAIN/`. This document is both the
architectural spec and the build plan.

---

## What Brain Is

`BRAIN/` is the layer between **memory** (facts retrieved from the graph) and
the **LLM** (the thing that generates words). Its job is to turn structured
context into a specific person speaking specifically to Zafar, right now.

Memory does not know about prompts, persona, emotional inference, or modes.
Brain does. Brain does not know about Neo4j, BM25, or spreading activation —
it only consumes the `WorkingContext` snapshot.

**Public surface — the only thing `sofi.py` imports:**
```python
from BRAIN.brain import Brain

brain = Brain()
await brain.setup()
async for token in brain.process(user_message):    # streaming from day 1
    ...
await brain.shutdown()
```

---

## Hard Constraints

| Constraint | Value | Reason |
|---|---|---|
| LLM provider | Groq, model `llama-3.3-70b-versatile` | Sub-second first-token, strong reasoning, 128k context |
| No LLM in routing / mode decision / state inference | Zero | Latency + determinism + debuggability |
| Streaming from day 1 | All response paths yield tokens | Voice-ready later, no rewrites |
| Per-turn brain code (excl. memory + LLM) | ≤ 30ms | Memory is ~1s; brain code must be invisible |
| Total token budget per prompt | ~2000–2300 tokens + message | Persona block ~900 tok; memory + state fill the rest |
| Personality consistency | Jarvis-hybrid SOFi every turn | No mode-induced split personality |
| Single LLM call per turn | Yes | Multi-call pipelines kill latency + voice continuity |
| Self-awareness | SOFi knows what subsystems she has | "Do you remember…" / "Can you…" must produce honest answers |
| Mode count | 4 (conversational, empathetic, focused, creative) | Sweet spot — meaningful distinctions without flicker |

---

## Architecture

```
sofi.py                          ← thin terminal loop (Rich-based UI)
  │
  ▼
BRAIN/brain.py                   ← single coordinator
  │
  ├── BRAIN/persona/             ← identity (already built)
  │     persona.py               5 mode-adjusted persona blocks, cached
  │     personality.json         identity, traits, speech rules, modes
  │
  ├── BRAIN/state/               ← user + sofi + self/meta state
  │     user_state.py            inferred emotional state, need, engagement
  │     sofi_state.py            mode, internal energy, current focus
  │     self_model.py            capabilities registry — what SOFi can do
  │
  ├── BRAIN/mode/                ← mode controller (non-LLM, dynamic)
  │     controller.py            scores all modes, applies hysteresis
  │     signals.py               extract signals from message + memory
  │
  ├── BRAIN/prompt/              ← turn structured data into prompt
  │     formatters.py            memory dicts → readable sentences
  │     builder.py               assemble final prompt (persona + state + memory + turns + msg)
  │
  ├── BRAIN/llm/                 ← Groq client wrapper
  │     groq_client.py           async streaming, retry, error handling
  │
  └── BRAIN/ui/                  ← terminal UI (Rich)
        cli.py                   prompt-toolkit input, Rich output, streaming render
        renderer.py              token-stream → live markdown panel
```

---

## Data Flow Per Turn

```
user types message in terminal
   │
   ▼
sofi.py reads line from prompt-toolkit
   │
   ▼
brain.process(message) starts:
   1.  await memory.observe("user", message)                # fires bg work, 0ms
   2.  ctx = await memory.get_context_async()               # ~800-1100ms (current)
   3.  user_state.infer(ctx, message)                       # < 1ms, rule-based
   4.  mode, allow_drop = mode_controller.decide(           # < 1ms, hysteretic
           ctx, user_state, prev_mode)                      #   allow_drop unlocks
   5.  sofi_state.update(mode, ctx)                         # < 1ms; logs into
                                                            #   ConversationLogger
   6.  prompt = builder.build(ctx, message, mode, allow_drop) # ~5-10ms
   7.  async for token in llm.stream(prompt):               # ~300-800ms first tok
        yield token
   8.  await memory.observe("assistant", full_response)     # 0ms
```

UI layer concurrently renders tokens to a Rich Live panel.

---

## Module Specs

### `BRAIN/persona/` — DONE (Jarvis-hybrid SOFi)
Built. `get_identity_block(mode, allow_dropped_formality=False)` returns a
Jarvis-coded persona block for one of 4 modes. Eight cached variants total
(4 modes × {standard, dropped-formality permitted}) via `warm_cache()`.

**Character (locked):**
- Jarvis chassis: composed, dry, slightly formal. Default address: 'sir' /
  'Mr. Zafar' / no address (most common — restraint is the voice).
- Hybrid mechanism — two earned exceptions:
  1. **Name-drop** — replaces 'sir' with 'Zafar' in genuinely weighty
     moments. ~1 in 20-30 turns. The model decides when to exercise it.
  2. **Dropped formality** — empathetic mode + emotional_intensity ≥ 0.6
     unlocks the permission via `allow_dropped_formality=True`. The model
     may drop 'sir' for 1-2 sentences. Permitted, not required.
- One-sentence default response length.
- Deadpan humor, sparing. Dry sarcasm pointed at situations more than at
  Zafar. Affection demonstrated, never stated.
- Behavioural never list locked in `personality.json` (12 items: no
  'Certainly!', no AI-disclaimers, no moralizing, etc.)

### `BRAIN/state/user_state.py` — DONE (Phase 3)
Pure-rule emotional + need inference. Inputs:
- `intent` from `ctx.memory.retrieval_meta`
- `emotional_baseline` (memory's emotion distribution)
- `emotional_tone` averaged over `must_know`
- Raw message text (regex on emotion keywords)
- Conversation tempo (turn length, gap)

Outputs written to `WorkingContext.user`:
- `current_emotional_state` ∈ {neutral, stressed, sad, excited, overwhelmed, focused, content, frustrated}
- `emotional_intensity` ∈ [0, 1]
- `current_need` ∈ {emotional_support, practical, informational, creative, casual}
- `engagement_level` ∈ {disengaged, normal, highly-engaged}

Runs < 1ms. Deterministic. No LLM.

### `BRAIN/state/sofi_state.py` — TO BUILD
SOFi's own state. Read by prompt builder, updated by mode controller.
- `current_mode` (the mode controller's decision)
- `emotional_tone` (warm | concerned | focused | playful | analytical)
- `current_focus` (what topic she's currently engaged with — pulled from active entities)
- `energy_level` (light proxy — high if recent emotional baseline is positive)

Brain layer keeps these honest. Not invented out of thin air.

### `BRAIN/state/self_model.py` — TO BUILD ← THE "UNIFIED" PIECE
What SOFi knows about herself. The reason she feels like one entity, not bolted-together subsystems.

```python
class SelfModel:
    capabilities: List[Capability]    # ["remember our conversations", "track ongoing topics", ...]
    limitations: List[str]            # ["can't access the web yet", "can't read your files yet"]
    current_subsystems: Dict          # {"memory": "active", "tools": "not yet", ...}

    def describe(self) -> str:
        """Short paragraph SOFi can reference when asked what she can do."""
```

Goes into prompt as a compact `━━━ WHAT YOU CAN DO RIGHT NOW ━━━` section
when the message looks like a meta-question ("can you…", "do you remember…",
"what do you know about yourself"). Otherwise omitted to save tokens.

### `BRAIN/mode/` — DONE (Phase 3)
Non-LLM, signal-scored, hysteretic. 4 modes: conversational, empathetic,
focused, creative. Inputs:
- `intent` (5 classes from memory)
- `user_state.current_need`
- `user_state.emotional_intensity`
- recent message text (regex signals for playfulness, urgency, technical depth)
- `previous_mode` (for hysteresis)

Algorithm:

```python
def decide(ctx, user_state, prev_mode) -> Mode:
    scores = {m: 0.0 for m in MODES}
    # 1. Intent-driven bias
    if intent == EMOTIONAL: scores[empathetic] += 0.6
    if intent == FACTUAL:   scores[focused] += 0.4
    if intent == ENTITY:    scores[conversational] += 0.3
    # 2. User need
    if need == emotional_support:  scores[empathetic] += 0.5
    if need == informational:      scores[focused] += 0.4
    if need == casual:             scores[conversational] += 0.5
    if need == creative:           scores[creative] += 0.5
    # 3. Emotional intensity
    if intensity > 0.4:            scores[empathetic] += 0.3
    # 4. Message-level signals (regex)
    if has_creative_signal(msg):   scores[creative] += 0.4
    if has_technical_signal(msg):  scores[focused] += 0.5
    if has_playful_signal(msg):    scores[conversational] += 0.3
    # 5. Hysteresis — staying in the same mode is slightly preferred
    scores[prev_mode] += 0.15
    # 6. Hard rules — high emotional intensity OVERRIDES everything → empathetic
    if user_state.intensity >= 0.7:
        return Mode.empathetic
    return max(scores, key=scores.get)
```

The hysteresis + override logic is what makes mode shifts feel intentional
rather than flickering. The controller also outputs the
`allow_dropped_formality` flag (True iff mode == empathetic AND intensity
≥ 0.6) which is forwarded to persona.get_identity_block().

### `BRAIN/prompt/formatters.py` — TO BUILD
Turn memory dicts into readable sentences. Strip internal fields. Add
emotional qualifier when |emotional_tone| > 0.4. Add time reference for
memories older than 7 days. Add social context from `participants`.

Cap each memory at 2 lines. Cap section at ~800 tokens.

### `BRAIN/prompt/builder.py` — TO BUILD
Assembles the final prompt. Layout:

```
━━━ WHO YOU ARE ━━━              persona.get_identity_block(mode)  ~700 tok
━━━ YOUR CHARACTER ━━━           (in persona block)
━━━ HOW YOU SPEAK ━━━            (in persona block)
━━━ HOW YOU ADAPT RIGHT NOW ━━━  (in persona block, mode-specific)

━━━ CURRENT MOMENT ━━━           time of day, weekday              ~30 tok
━━━ WHAT'S TRUE FOR ZAFAR ━━━    user_state (emotional, need)      ~60 tok
━━━ WHAT YOU CAN DO ━━━          self_model.describe() — IFF meta  ~80 tok (conditional)

━━━ WHAT YOU REMEMBER ━━━        formatted must_know               ~300 tok
━━━ BACKGROUND CONTEXT ━━━       formatted context                 ~300 tok
━━━ LOOSELY RELATED ━━━          formatted associations (optional) ~150 tok

━━━ RECENT CONVERSATION ━━━      last 5 turns                      ~250 tok

━━━ ZAFAR JUST SAID ━━━          current message verbatim          variable

━━━ RESPOND AS SOFI ━━━          mode closing instruction          (in persona)
```

Total: ~1800 tokens + message. Sections that have no content are omitted.

### `BRAIN/llm/groq_client.py` — TO BUILD
Async streaming Groq client. Returns `AsyncIterator[str]` of tokens.
Retry-with-backoff on transient failures. Honest error surfacing — never
silently return "Sorry, something went wrong."

### `BRAIN/ui/cli.py` — TO BUILD
Terminal UI. Library stack (locked): **`rich` + `textual` (hybrid) +
`prompt_toolkit`**.
- `rich` — markdown rendering, syntax highlighting, live token streaming,
  status panels. Fast iteration.
- `textual` — used selectively for richer layout (split panes, status bar)
  if/when V1 polish requires it. Same author as `rich`, fully compatible.
- `prompt_toolkit` — input handling with history (up arrow), multi-line
  input (Shift-Enter), key bindings, autocomplete for slash commands.

This is the Python equivalent of the Ink (JS/React) stack used by Gemini
CLI and Claude Code.

Features for V1:
- Streaming token render with Rich `Live` + markdown
- prompt_toolkit input with history + multi-line
- Status line: current mode, retrieval latency, token count
- Slash commands: `/mode`, `/clear`, `/save`, `/help`, `/exit`

### `BRAIN/brain.py` — TO BUILD
Single coordinator class. Owns the MemoryManager, the LLM client, the
mode controller. Provides `process(message) -> AsyncIterator[str]`.

### `sofi.py` — TO BUILD
Thin entry point at workspace root. Calls `brain.process()`, streams to
`ui.cli`. ~50 lines.

---

## Mode Controller — Why Non-LLM Can Still Be Advanced

The naive worry: rules feel rigid. Three things keep ours dynamic:

1. **Continuous scoring, not branching** — every mode gets a score from
   multiple signals. The winner emerges from accumulated evidence rather
   than a single if/else.

2. **Hysteresis** — staying in the same mode gets a +0.15 bonus. Prevents
   thrashing between modes turn-by-turn when scores are close.

3. **Hard overrides** — emotional intensity ≥ 0.7 forces empathetic mode.
   No exceptions. These act as safety floors so the system never fails
   the user in a moment of distress.

4. **Signal pluralism** — we read intent, need, intensity, message regex
   (creative/technical/playful), and previous mode. Five independent
   signal sources gives genuinely emergent behaviour.

Total cost: ~0.1ms per turn. Fully deterministic, fully debuggable.

---

## Token Budget (Per Turn)

| Section | Tokens | Conditional? |
|---|---|---|
| Persona block (mode-adjusted) | ~900 | Always (~1000 if dropped-formality permitted) |
| Current moment | ~30 | Always |
| User state | ~60 | Always |
| Self model | ~80 | Only on meta-questions |
| Must-know memories (≤ 5) | ~300 | Skipped if empty |
| Background context (≤ 10) | ~300 | Skipped if empty |
| Loose associations (≤ 10) | ~150 | Skipped if empty |
| Recent turns (5) | ~250 | Skipped on first turn |
| User message | variable | Always |
| **Total** | **~2000–2300** | + message |

Groq `llama-3.3-70b-versatile` supports 128k context. We're using ~1.5%
of it. Plenty of headroom for memory growth.

---

## Build Order (Phased)

### Phase 1 — Skeleton + persona-only chat (1 day)
- `BRAIN/llm/groq_client.py`
- `BRAIN/ui/cli.py` (basic Rich + prompt_toolkit)
- `BRAIN/brain.py` (no memory, no state, no modes — just persona + LLM)
- `sofi.py`
- **Acceptance:** can have a text chat with SOFi using persona-only prompt.

### Phase 2 — Wire memory + state (1 day)
- `BRAIN/state/user_state.py`
- `BRAIN/state/sofi_state.py`
- `BRAIN/state/self_model.py`
- Wire `MemoryManager` into `brain.py`
- **Acceptance:** SOFi remembers across turns, knows what she can do.

### Phase 3 — Mode controller (half day)
- `BRAIN/mode/signals.py`
- `BRAIN/mode/controller.py`
- **Acceptance:** SOFi shifts tone correctly when user shifts mood; no
  flickering.

### Phase 4 — Prompt assembly + memory formatting (1 day)
- `BRAIN/prompt/formatters.py`
- `BRAIN/prompt/builder.py`
- **Acceptance:** memories surface as natural references, not bullet lists.

### Phase 5 — Polish (1 day)
- Slash commands in CLI
- Status line (mode, latency, tokens)
- Robust error handling (LLM failure, Neo4j down, etc.)
- **Acceptance:** can use SOFi as daily driver for text chat.

**Total: ~5 days of focused work.**

---

## Acceptance Criteria (V1 — End of Phase 5)

A 20-message conversation where:

1. SOFi remembers details across turns (memory wired correctly)
2. She shifts tone when emotional state shifts (mode controller works)
3. She doesn't flicker between modes mid-conversation (hysteresis works)
4. When asked "what can you do?", her answer matches her actual capabilities (self_model works)
5. When asked "do you remember X?", she answers honestly based on retrieval (no hallucinated recall)
6. Responses feel like *one* SOFi across modes — not a different AI each turn
7. Time-to-first-token < 2s warm, end-to-end response < 4s for typical messages
8. Terminal UI renders cleanly with streaming, no flicker
9. Token budget stays under 2500 per turn

---

## What's Deliberately Excluded From V1

These are real things but they're not Phase 1-5:

- **STT / TTS** — separate work, after V1 (user has plans here)
- **Tool calling** (calendar, web, files) — V2
- **Proactivity** (SOFi speaks first) — V2; wire to WorkspaceWatcher then
- **Multi-user** — SOFi is Zafar-specific by design
- **Multi-modal** — text in, text out for V1
- **Streaming TTS pipeline** — voice-only concern

---

## Locked Decisions (2026-05-21)

| Question | Decision |
|---|---|
| Groq model | `llama-3.3-70b-versatile` |
| Mode count | 4: conversational, empathetic, focused, creative |
| Persona direction | Jarvis-hybrid (formal chassis, two earned exceptions) |
| Address default | "sir" / "Mr. Zafar" / no address (most common) |
| Name exception | "Zafar" — ~1 in 20-30 turns, weighty moments only |
| Dropped formality | Empathetic mode + intensity ≥ 0.6 unlocks it |
| Behavioural never list | 12 items, locked in `personality.json` |
| SOFi self-state memory | Option B — log `sofi_mode` + `sofi_emotional_tone` per turn |
| Conversation session handling | Already in `ConversationLogger` — UUID + 30-min timeout |
| CLI stack | `rich` + `textual` (hybrid) + `prompt_toolkit` |
| Default response length | One sentence; expand only when depth earned |

## Still Open (Discuss During Build)

- Things I flagged earlier: cold-start greeting, how she says "I don't know", handling empty memory, matching response length to input length. To revisit in Phase 1 once we see real conversation behaviour.

---

## Reading Order For New Code Reviewers

1. `personality.json` — the source of SOFi's voice
2. `persona/persona.py` — how the persona block is built
3. This file — architecture
4. `brain.py` (when built) — the entry point
5. `state/`, `mode/`, `prompt/` — the pieces that compose into brain.py
