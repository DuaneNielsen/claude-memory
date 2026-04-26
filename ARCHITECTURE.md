# Claude Memory System — Architecture

## Overview

A local memory system that indexes Claude Code conversation history into searchable atomic facts, exposed back to Claude Code via MCP so it can query past conversations automatically.

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION (batch, one-time + incremental)│
│                                                             │
│  ~/.claude/projects/*/*.jsonl                               │
│       │                                                     │
│       ▼                                                     │
│  Session Parser ──► Turn Extraction ──► EDU Extraction      │
│  (parse JSONL,      (user/assistant     (LLM decomposes     │
│   extract msgs,      messages with       into atomic         │
│   resolve meta)      timestamps)         self-contained      │
│                                          facts)              │
│       │                                                     │
│       ▼                                                     │
│  Embedding ──► ChromaDB                                     │
│  (nomic-embed-text    (vector store +                       │
│   or bge-small)        metadata index)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     RETRIEVAL (query-time, via MCP)          │
│                                                             │
│  Claude Code ──► MCP Server ──► Query Pipeline              │
│                                    │                        │
│                                    ├─ Dense similarity      │
│                                    ├─ Project/topic filter   │
│                                    ├─ Recency weighting     │
│                                    │                        │
│                                    ▼                        │
│                              Top-K EDUs ──► LLM Filter      │
│                              (candidates)    (relevance     │
│                                               pruning)      │
│                                    │                        │
│                                    ▼                        │
│                              Return to Claude Code          │
│                              (< 1K tokens typically)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Session Parser (`parser.py`)

Reads Claude Code JSONL session files. Each line is a message with:
- `type`: "user" | "assistant" | "file-history-snapshot" | etc
- `message.role`: "user" | "assistant"  
- `message.content`: string or list of content blocks (text, tool_use, tool_result)
- `timestamp`: ISO 8601
- `cwd` / `sessionId` / `gitBranch` / `version`
- `parentUuid` / `uuid`: message tree structure

**Key decisions:**
- Skip `file-history-snapshot`, `tool_result` content (file diffs, command output — too noisy)
- Extract user prompts and assistant text responses only
- Preserve timestamps, project path (derive from parent dir name), session ID
- Group into conversation turns (user message + assistant response)
- Handle content blocks: extract `text` type, skip `tool_use`/`tool_result` blocks

**Output:** List of `Turn` objects:
```python
@dataclass
class Turn:
    turn_id: int
    session_id: str
    project: str          # e.g. "primesignal", "frigate"
    timestamp: datetime
    speaker: str          # "user" or "assistant"
    text: str             # extracted text content
    git_branch: str | None
```

### 2. EDU Extractor (`extractor.py`)

Sends conversation turns to a local LLM (Qwen3.5-27B via llama-server or any OpenAI-compatible endpoint) to decompose into Elementary Discourse Units.

**Key decisions from papers:**
- Process one session at a time (natural conversation boundary)
- Include both user and assistant turns — user messages contain intent/preferences, assistant messages contain decisions/facts
- One-shot prompting with a worked example (EMem approach)
- Output as structured JSON with source turn IDs for traceability
- Batch sessions to amortize LLM startup cost

**Output:** List of `EDU` objects:
```python
@dataclass
class EDU:
    edu_id: str           # UUID
    text: str             # self-contained atomic fact
    source_turn_ids: list[int]
    session_id: str
    project: str
    timestamp: datetime   # from earliest source turn
    speakers: list[str]   # who was involved
```

### 3. Embedding + Storage (`store.py`)

Embeds EDUs and stores in ChromaDB with metadata.

**Embedding model options (ranked by practical fit):**
- `nomic-embed-text-v1.5` — 768d, 8192 token context, ~275MB. Best quality/size ratio
- `bge-small-en-v1.5` — 384d, 512 token context, ~130MB. Faster, smaller
- `all-MiniLM-L6-v2` — 384d, 256 token context, ~80MB. Smallest, sentence-transformers native

Run on GPU (trivial VRAM) or CPU (fast enough for <100K EDUs).

**ChromaDB collection schema:**
```python
collection.add(
    ids=[edu.edu_id],
    documents=[edu.text],
    embeddings=[embedding],
    metadatas=[{
        "session_id": edu.session_id,
        "project": edu.project,
        "timestamp": edu.timestamp.isoformat(),
        "speakers": ",".join(edu.speakers),
        "source_turns": json.dumps(edu.source_turn_ids),
    }]
)
```

**Key decisions:**
- ChromaDB over Qdrant — zero-config, SQLite-backed, sufficient for this scale
- Store EDU text as the document (what gets embedded and searched)
- Metadata enables filtering by project, time range, etc.
- Persist to disk (`~/.local/share/claude-memory/chromadb/`)

### 4. Query Pipeline (`query.py`)

Multi-signal retrieval with LLM filtering.

**Retrieval stages:**
1. **Dense search**: Embed query, find top-30 similar EDUs from ChromaDB
2. **Metadata filter**: Optionally scope to project, time range
3. **Recency weighting**: Apply exponential decay to similarity scores
   - `final_score = similarity * e^(-α * days_ago)` where α ≈ 0.005-0.01
   - Recent facts get mild boost, very old facts get mild penalty
   - NOT aggressive — old facts should still surface if highly relevant
4. **LLM filter** (optional): Send top-15 candidates + query to LLM, ask it to select the relevant ones
   - Use the EMem approach: "Be MAXIMALLY INCLUSIVE — prefer false positives over false negatives"
   - This step costs ~500 tokens but significantly improves precision
5. **Return top-K** (typically 5-10 EDUs, <1K tokens total)

### 5. MCP Server (`server.py`)

Exposes the query pipeline as an MCP tool that Claude Code can call.

**Tool definition:**
```json
{
    "name": "search_conversation_memory",
    "description": "Search past Claude Code conversations for relevant context. Use this when the user references past work, asks 'did we discuss...', or when historical context would help with the current task.",
    "parameters": {
        "query": "Natural language search query",
        "project": "Optional: filter to a specific project name",
        "time_range": "Optional: 'last_week', 'last_month', 'last_3_months', or 'all'",
        "max_results": "Optional: number of results (default 10)"
    }
}
```

**Server implementation:**
- stdio-based MCP server (standard for Claude Code)
- Loads ChromaDB + embedding model on startup
- Stateless query handling
- Register in `~/.claude/mcp.json`

### 6. Incremental Ingestion (`ingest.py`)

After initial bulk ingestion, new sessions need to be indexed.

**Approach:**
- Track ingested sessions in a metadata table (session_id → last_ingested_line)
- On each run, scan for new/updated JSONL files
- Process only new content
- Can be triggered manually or via a hook/cron

### 7. Project Index (`index_builder.py`)

Per-project markdown digest, written to `~/.local/share/claude-memory/indices/<project>.md` and injected at SessionStart by the `session_start_index.py` hook. One file per project.

**Catalog, not digest.** The index advertises *what memories exist* for the project so Claude knows what to ask for via `recall_get_context`. It does NOT pre-load the content of those memories into the system prompt — that would be wasted tokens (most aren't relevant to the upcoming conversation) and would prime Claude with potentially-superseded facts. The actual EDU text lives in ChromaDB and is fetched on demand.

The one exception is the **Preferences** section: those shape behavior immediately and can't be deferred to a tool call, so the full text is included verbatim.

**Section content vs reference cut:**

| Section | Type | Why |
|---|---|---|
| `project_overview` | content (~2 sentences) | Claude needs to orient before any tool call |
| `preferences` | content (full EDU text) | Shape behavior immediately; cannot be deferred |
| `available_memories` | reference (counts + topics) | "23 decisions tagged: pipewire, niri, …" — Claude calls `recall_get_context` if a topic looks relevant |
| `active_threads` | reference (one-line summaries) | Pointers to ongoing trajectories |
| `recent_activity` | reference (one-line trajectory summaries) | Already a catalog format |
| `keyword_cloud` | reference (keywords only) | Always present — also the topic-coverage hint |

**Two builders:**
1. **`curate_index_with_llm` (Opus, primary)** — sees a wide slice of the project's distilled data (trajectory digest, all preference EDUs verbatim, decision/gotcha/architecture EDUs capped at 80/tag, keyword frequencies, full per-tag counts) and emits a structured `CuratedIndex` that the renderer turns into markdown. Catches superseded preferences, writes a `project_overview`, and groups the per-tag EDUs into topic clusters.
2. **`_build_index_deterministic` (no LLM, fallback)** — the original keyword-frequency-ranked builder. Runs unconditionally if the LLM call fails (rate-limit, missing CLI, malformed JSON), so the system stays usable offline.

**Hash gating.** `write_index` hashes the `IndexInput` (sorted trajectory IDs + summaries + EDU IDs + texts + per-tag counts). If the hash matches `<project>.md.meta.json`, no LLM call fires — the existing index file is reused as-is. The Opus call only runs when something genuinely new arrived. Per-call cost: ~$0.05–0.15 with input ~6–12k tokens, output ~250–400 tokens.

**Token budget.** The injected wrapper text is byte-stable so prompt-cache hits still work for the framing; only the body changes when the index rebuilds. `enforce_token_budget_curated` defensively trims sections in priority order (`keyword_cloud → recent_activity → available_memories → active_threads → preferences → project_overview`) until the body fits `MAX_INDEX_TOKENS = 600`. The catalog framing typically lands well under that.

## Directory Structure

```
claude-memory/
├── ARCHITECTURE.md          # this file
├── PLAN.md                  # implementation plan
├── PROMPTS.md               # all LLM prompts
├── PAPERS.md                # key implementation details from papers
├── pyproject.toml           # uv project config
├── src/
│   └── claude_memory/
│       ├── __init__.py
│       ├── parser.py        # JSONL session parsing
│       ├── extractor.py     # EDU extraction via LLM
│       ├── store.py         # ChromaDB embedding + storage
│       ├── query.py         # multi-signal retrieval pipeline
│       ├── server.py        # MCP server
│       ├── ingest.py        # bulk + incremental ingestion CLI
│       └── config.py        # paths, model endpoints, parameters
└── tests/
    ├── test_parser.py
    ├── test_extractor.py
    └── test_query.py
```

## Resource Requirements

- **VRAM**: ~275MB for nomic-embed-text (trivial). Qwen3.5-27B for extraction uses ~17.4GB but only needed during ingestion, not at query time (unless LLM filter is enabled)
- **Disk**: ChromaDB index for ~50K EDUs ≈ 200-500MB
- **RAM**: ChromaDB loads index into memory, ~500MB-1GB for this scale
- **Ingestion time**: ~90MB of sessions, estimate 5-15 min with local LLM (bottleneck is EDU extraction)
- **Query latency**: <500ms for embedding + ChromaDB search. +2-5s if LLM filter is used.

## Key Design Principles (from papers)

1. **Decompose first, search second** — quality of EDU extraction determines everything downstream
2. **Self-contained facts** — each EDU must be understandable without context. Resolve pronouns, infer dates, include entity names
3. **Recency decay, not recency cutoff** — old facts should still be findable, just mildly deprioritized
4. **Maximally inclusive retrieval** — false positives are cheap (LLM ignores irrelevant context), false negatives are expensive (lost information)
5. **Lightweight wins** — EMem without graph nearly matches EMem with PageRank. Skip the graph unless retrieval quality proves insufficient

---

## North Star: Ambient Associative Memory (future direction)

The current design is **query-based retrieval** — something asks, memory answers.
Human memory doesn't work that way. Memories *surface* associatively, cued by
current context, without explicit querying. You see Alice and memories of Alice
appear — you don't formulate a query.

The long-term direction for claude-memory is **ambient associative memory**:

- **Three layers**:
  1. *Ambient* — always-on monitor, surfaces memories as events when context cues hit
  2. *Associative* — classifier fires on explicit references ("that thing we did")
  3. *Explicit* — deliberate subagent query (current design — stays as fallback)

- **Cue-based, not query-based** — extract entities/topics/intents from recent turns,
  use those as implicit queries. The conversation *is* the query.

- **Salience filtering** — only high-confidence matches surface, rest stay dormant.
  Salience = similarity × recency × importance × novelty-vs-current-context.

- **Event-driven surfacing** — memories appear *unprompted* when triggered,
  injected as annotated events ("memory: you configured X at port Y last week").

- **Novelty / dedup tracking** — never resurface memories already in context this
  session. Humans don't re-remind themselves of things they just thought about.

**Why not build this now**: walking the query-based path first provides the
benchmark-grounded intuitions needed to design the surfacing layer. Threshold,
salience formula, and cue extraction all need empirical tuning that only makes
sense once the retrieval itself is measured and tuned. Ambient memory built on
a weak retrieval substrate just surfaces noise more aggressively.

**Convergence target**: the ambient layer becomes the primary interaction mode,
the explicit subagent / MCP tool becomes a fallback for complex multi-hop queries
the ambient layer can't resolve. Human memory works the same way — you mostly
don't query, but you *can* when you need to.

