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
