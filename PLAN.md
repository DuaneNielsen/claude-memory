# Claude Memory System — Implementation Plan

## Phase 1: Foundation (get something working end-to-end)

### 1.1 Project Setup
- Init uv project with Python 3.13
- Dependencies: `chromadb`, `sentence-transformers` (or `fastembed`), `mcp` (python MCP SDK), `pydantic`, `httpx` (for llama-server API calls)
- Config: model endpoint, ChromaDB path, session paths, embedding model name

### 1.2 Session Parser
- Read all `~/.claude/projects/*/*.jsonl` files (96 sessions, ~90MB)
- Parse each JSONL line, filter to `type: "user"` and `type: "assistant"` with text content
- Handle content block format: extract `text` blocks, skip `tool_use` and `tool_result`
- Extract metadata: timestamp, session_id, project (from parent dir), git_branch
- Assign sequential turn IDs within each session
- Output: list of Turn dataclasses per session

**Test**: Parse one known session, verify turn count and text extraction.

### 1.3 EDU Extractor
- Connect to local LLM (OpenAI-compatible endpoint — llama-server or Unsloth proxy)
- For each session, format turns into the extraction prompt (system + one-shot + session text)
- Parse JSON response into EDU dataclasses
- Handle long sessions: if > 50 turns, split into chunks of 30 turns with 5-turn overlap, then deduplicate EDUs across chunks
- Rate limit / retry logic for LLM calls
- Progress tracking (sessions can take 10-30s each)

**Test**: Extract EDUs from 3 diverse sessions, manually verify quality.

### 1.4 ChromaDB Storage
- Initialize persistent ChromaDB collection at `~/.local/share/claude-memory/chromadb/`
- Load embedding model (start with `all-MiniLM-L6-v2` for speed, upgrade later if needed)
- Embed + store all EDUs with metadata
- Track ingestion state: `ingested_sessions.json` mapping session_id → {file_path, line_count, edu_count, timestamp}

**Test**: Store 100 EDUs, run similarity queries, verify results make sense.

### 1.5 Basic Query
- Embed query string
- ChromaDB similarity search, top-30
- Apply recency decay: `score = similarity * exp(-0.007 * days_ago)`
- Return top-10 by adjusted score
- Print results to stdout for testing

**Test**: Query "audio problems" and verify PipeWire/EasyEffects EDUs rank high.

### 1.6 MCP Server (minimal)
- stdio MCP server with single `search_conversation_memory` tool
- Takes query string + optional project filter + optional max_results
- Returns formatted list of EDUs with timestamps and project names
- Register in `~/.claude/mcp.json`

**Test**: Start Claude Code, ask about past work, verify it calls the MCP tool and gets useful results.

**Phase 1 deliverable**: Working end-to-end pipeline. Claude Code can search past conversations.

---

## Phase 2: Quality (make retrieval actually good)

### 2.1 Prompt Iteration
- Run EDU extraction on 5-10 sessions, review output manually
- Tune the extraction prompt based on common failure modes:
  - Too granular (splitting one fact into fragments)?
  - Too coarse (merging distinct facts)?
  - Missing technical details (paths, configs)?
  - Bad pronoun resolution?
  - Missing temporal context?
- Update one-shot example to cover edge cases seen in real data

### 2.2 Evaluation Set
- Create 50 questions about past conversations with known answers
- Manually label which EDUs should be retrieved for each question
- Measure Recall@10 and Recall@20
- Track accuracy over time as prompts/pipeline evolves

### 2.3 LLM Relevance Filter
- Add optional LLM filter stage at query time
- Send top-15 candidate EDUs + query to local LLM
- LLM selects relevant subset (maximally inclusive)
- Measure impact on accuracy vs latency tradeoff
- Make it configurable (skip for speed, enable for complex queries)

### 2.4 Embedding Model Upgrade
- Benchmark `all-MiniLM-L6-v2` vs `nomic-embed-text-v1.5` vs `bge-small-en-v1.5` on evaluation set
- Re-embed entire collection with winning model if different
- Test on GPU vs CPU performance

### 2.5 Session Summaries
- Generate 1-2 sentence summary per session from its EDUs
- Store as separate ChromaDB collection or metadata
- Enable "what did we work on last week?" type queries
- Coarse-grained index for browsing

---

## Phase 3: Polish (incremental updates, robustness)

### 3.1 Incremental Ingestion
- Watch for new/updated JSONL files since last run
- Process only new content (track line counts per file)
- CLI command: `claude-memory ingest` (bulk) and `claude-memory update` (incremental)
- Could hook into Claude Code session end event via hooks

### 3.2 Deduplication
- When new EDUs are similar (>0.95 cosine) to existing ones, flag as potential duplicates
- Keep the more recent/complete version
- Log dedup decisions for review

### 3.3 Metadata Enrichment
- Auto-tag EDUs with topic categories (audio, networking, projects, config, etc.)
- Extract project names more reliably from session context
- Add "importance" scoring (how likely is this fact to be useful later?)

### 3.4 Query Improvements
- Query expansion for vague queries (generate sub-queries via LLM)
- Project-scoped search (auto-detect current project, bias results)
- Time-range filters ("last month", "before March")

### 3.5 CLI Tools
- `claude-memory stats` — collection size, EDU count, session count, storage size
- `claude-memory search "query"` — CLI search for testing
- `claude-memory sessions` — list ingested sessions
- `claude-memory reingest <session_id>` — re-extract EDUs for a session

---

## Phase 4: Future (if the system proves useful)

- Cross-session entity resolution (merge "primesignal" / "the trading project" / "primesignal project")
- Importance scoring with Generative Agents-style 1-10 rating at extraction time
- Multi-user support (if others use Claude Code on this machine)
- Web UI for browsing/searching memory
- Memory from other sources (terminal history, git commits, notes)
- Contradiction detection (new fact conflicts with existing fact → flag or auto-resolve)

---

## Technology Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.13 | Ecosystem (chromadb, sentence-transformers, mcp SDK) |
| Package manager | uv | Already used for other projects |
| Vector DB | ChromaDB | Zero-config, SQLite-backed, sufficient for <100K docs |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2 → nomic-embed-text) | Local, GPU/CPU, no API dependency |
| LLM (extraction) | Qwen3.5-27B via llama-server | Already running locally, sufficient quality |
| LLM (query filter) | Same, or skip for speed | Optional component |
| MCP framework | mcp Python SDK | Standard for Claude Code integration |
| Data format | Pydantic models | Type-safe parsing of LLM output |

## Key Parameters (from papers)

| Parameter | Value | Source |
|-----------|-------|--------|
| Recency decay α | 0.005-0.01 per day | Memoria (adapted from per-minute to per-day) |
| Retrieval top-K (candidates) | 20-30 | EMem uses 30, Memoria uses 20 |
| Final results returned | 5-10 | EMem averages 738 tokens ≈ ~10 EDUs |
| Max context to caller | < 1000 tokens | EMem/Memoria convergent finding |
| Embedding dimension | 384-768 | MiniLM=384, nomic=768 |
| ChromaDB distance metric | cosine | Standard for text similarity |
| Session chunk size | 30 turns | Practical limit for extraction quality |
| Chunk overlap | 5 turns | Continuity across chunks |
