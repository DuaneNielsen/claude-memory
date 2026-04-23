# Plan: Trajectory-Based Memory with Session-Start Index + Subagent Retrieval

A rework of claude-memory around **topic trajectories** as the primary
structure, with a small session-start index for orientation and an Opus
subagent for deep retrieval.

## What changes vs. the original plan

The original plan stored isolated EDUs and tried to surface them via a
pre-computed index. Problem: EDUs in isolation lose conversational
context — "we decided X" is meaningless without the surrounding why.

New architecture:

1. **Trajectories** — contiguous turn-ranges covering a single topic.
   Each trajectory has its own EDUs, keywords, and 1-line summary.
2. **Hierarchical memory** — trajectory summaries are the "chapters",
   EDUs are the "paragraphs". Index shows chapters; retrieval returns
   chapter-sized windows.
3. **Subagent retrieval** — an Opus subagent handles deep reads
   (neighbor expansion, re-stitching, synthesis) so the main agent's
   context stays clean.

## Architecture Overview

```
                     ┌──────────────────────────────┐
                     │     SessionStart Hook        │
                     │  (injects ~500-token index)  │
                     └──────────────┬───────────────┘
                                    │ reads
                                    ▼
                     ┌──────────────────────────────┐
                     │ indices/{project}.md (file)  │
                     │  - Preferences               │
                     │  - Recent Activity (summaries)│
                     │  - Key Decisions/Gotchas     │
                     │  - Keyword Cloud             │
                     └──────────────────────────────┘
                                    ▲ rebuilt on
                                    │ ingestion
           ┌────────────────────────┴────────────────────────┐
           │                                                 │
┌──────────────────────┐                          ┌─────────────────────┐
│  ChromaDB `memories` │                          │ SQLite side DB      │
│  (EDUs, embeddings)  │◄───trajectory_id────────►│  trajectories       │
│                      │                          │  trajectory_keywords│
└──────────────────────┘                          └─────────────────────┘
           ▲                                                 ▲
           │                                                 │
           └─────────────┬───────────────────────────────────┘
                         │ queried by
                         ▼
                ┌────────────────────┐
                │  recall_memory()   │  ← MCP tool exposed to main agent
                │   MCP tool wraps   │
                │  Opus subagent     │
                └────────────────────┘
```

## Data Model

### EDU (ChromaDB `memories` collection)

Existing fields:
- `edu_id`, `text`, `source_turn_ids`, `session_id`, `project`,
  `timestamp`, `speakers`, `tag` (Step 1 — already landed)

New fields:
- `trajectory_id: str` — UUID, links to SQLite trajectories row
- `trajectory_index: int` — 0..N-1 position within trajectory

### Trajectory (SQLite)

```sql
CREATE TABLE trajectories (
    id              TEXT PRIMARY KEY,         -- UUID
    session_id      TEXT NOT NULL,
    project         TEXT NOT NULL,
    start_turn      INTEGER NOT NULL,
    end_turn        INTEGER NOT NULL,
    summary         TEXT NOT NULL,             -- 1-line LLM summary
    created_at      TEXT NOT NULL              -- ISO timestamp
);

CREATE INDEX idx_traj_session ON trajectories(session_id);
CREATE INDEX idx_traj_project ON trajectories(project);

CREATE TABLE trajectory_keywords (
    trajectory_id   TEXT NOT NULL,
    keyword         TEXT NOT NULL,
    PRIMARY KEY (trajectory_id, keyword),
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id) ON DELETE CASCADE
);

CREATE INDEX idx_kw_keyword ON trajectory_keywords(keyword);
CREATE INDEX idx_kw_project ON trajectory_keywords(trajectory_id);
```

**Why SQLite and not a second ChromaDB collection:** trajectory lookup
is keyword-exact-match and join-heavy. SQL is the right tool.

### Canonical Keyword Cloud

`SELECT DISTINCT kw.keyword FROM trajectory_keywords kw JOIN trajectories t ON kw.trajectory_id = t.id WHERE t.project = ?`

Scoped per-project. "pipewire" in `home` ≠ "pipewire" in some other
context.

## Extraction Pipeline (Option B: segment → extract together)

Single LLM call per session chunk produces **everything** in one pass:

**Input:**
- Session turns (chunk)
- Current project keyword cloud (if any)
- Existing trajectories for this session (if incremental)

**Output schema:**
```json
{
  "trajectories": [
    {
      "start_turn": 1,
      "end_turn": 7,
      "summary": "Replaced EasyEffects with native PipeWire filter-chain for mic processing",
      "keywords": ["pipewire", "audio", "easyeffects"],
      "edus": [
        {"text": "...", "source_turn_ids": [1, 3], "tag": "decision"},
        {"text": "...", "source_turn_ids": [4], "tag": "config"}
      ]
    },
    {
      "start_turn": 8,
      "end_turn": 12,
      "summary": "Configured dual-source transcriber to capture mic and system audio",
      "keywords": ["transcriber", "pipewire"],
      "edus": [...]
    }
  ]
}
```

**Key prompt rules:**
- Trajectories must be **contiguous** and **cover all substantive turns**
  (skipping filler is fine)
- Keywords: **prefer existing keywords from the provided cloud**. Only
  introduce new keywords when no existing one fits.
- Keywords are lowercase, short (1-3 words typically), canonical
  (prefer "pipewire" over "PipeWire" or "pipewire-0.3")
- Summary: single sentence, past tense, action-oriented

### Canonical Keyword Normalization (post-extraction)

For each *new* keyword (one not in the existing cloud):
1. Compute embedding similarity vs. existing keywords in the same project
2. If max similarity ≥ 0.9 → auto-merge to the existing canonical form
3. Else: accept as new canonical keyword
4. Flag ambiguous cases (similarity 0.75-0.9) in a log for periodic review

Cheap: embed the keyword string (same SentenceTransformer already loaded).

## Index Format

Injected at session start, pre-computed at ingestion time. Target
400-600 tokens. Project-scoped.

```markdown
## Preferences
- Casual, direct tone; no corporate filler
- Don't mock the database in integration tests
- Process management: one step at a time (check/kill/verify/start)

## Recent Activity (this project)
- [2026-04-21] Reworked memory architecture around trajectories + subagent retrieval
- [2026-04-20] Benchmarked model comparison for EDU extraction
- [2026-04-19] Designed session-start index injection hook
- [2026-04-13] Switched claude-memory to ChromaDB HTTP client-server mode
- [2026-04-13] Added plugin marketplace manifest

## Key Decisions / Gotchas
- ChromaDB HTTP client-server (not in-process) — avoids SQLite lock contention
- EDU extraction uses claude CLI with --json-schema enforcement
- Trajectories are contiguous turn-ranges, LLM decides boundaries
- Keyword cloud is project-scoped; canonical via embedding-similarity merge

## Keyword Cloud
chromadb, claude-memory, edu, extractor, ingestion, mcp, memory-index,
opus, pipewire, ranking, retrieval, sessionstart, subagent, trajectory
```

### Section ranking rules

- **Preferences:** all EDUs with `tag=PREFERENCE`, dedup
- **Recent Activity:** top ~5-8 *trajectories* by recency, showing summary
- **Key Decisions / Gotchas:** top ~5-8 *EDUs* with `tag ∈ {decision, gotcha}`
  ranked by `access_count × log(days_since_creation + 1)`
- **Keyword Cloud:** all distinct keywords for the project, alphabetized

## Retrieval Pipeline (Subagent)

Exposed as MCP tool `recall_memory(search_terms, question)`. Internally:

1. **Parse inputs**: search_terms (list), question (the thing the main
   agent is trying to answer)

2. **Spawn Opus subagent** with access to:
   - `search_trajectories_by_keyword(keyword)` — SQLite lookup
   - `search_trajectories_by_vector(query_text, n=20)` — ChromaDB vector
     search, returns trajectory_ids of hit EDUs
   - `fetch_trajectory_window(trajectory_id, pad=5)` — returns EDUs in
     [start - pad, end + pad] ordered by session turn

3. **Subagent strategy** (prompted):
   - For each search term, call keyword lookup → set of trajectory_ids
   - Also call vector search on the question → set of trajectory_ids
   - Union all hit trajectory_ids
   - For each trajectory, fetch window with N=5 neighbors
   - Re-stitch overlapping windows (collapse if ranges overlap)
   - Budget: stop adding trajectories once recovered context hits ~30k
     tokens (plenty of headroom in Opus 1M)
   - Synthesize answer to the main agent's question from the wall of
     recovered context
   - Return a focused response (~200-500 tokens typically)

4. **Return to main agent**: synthesized answer, not raw EDUs.

## Implementation Steps

### ✅ Step 1 — Semantic tags on EDUs (DONE)
Landed 2026-04-21. Six-value EDUTag enum drives section grouping in the
index. Orthogonal to keywords.

### Step 2 — Trajectory data model

- New module `src/claude_memory/trajectories.py`
- SQLite DB at `~/.local/share/claude-memory/trajectories.db`
- Tables per schema above
- `TrajectoryStore` class with methods:
  - `add_trajectory(t: Trajectory) -> None`
  - `get_by_id(traj_id: str) -> Trajectory | None`
  - `get_by_session(session_id: str) -> list[Trajectory]`
  - `get_keywords_for_project(project: str) -> list[str]`
  - `search_by_keyword(project: str, keywords: list[str]) -> list[str]`
  - `delete_session(session_id: str) -> int`
- Add `trajectory_id` and `trajectory_index` to ChromaDB EDU metadata

**Effort:** 3-4 hours

### Step 3 — Rework extractor for trajectory-mode extraction

- New `TRAJECTORY_SYSTEM_PROMPT` that produces the nested JSON schema
- New one-shot example (reuse the EasyEffects/PipeWire conversation but
  segmented into 2-3 trajectories)
- `extract_trajectories_from_session(session, keyword_cloud, existing_trajectories=[])`
  returns `list[Trajectory]` with attached EDUs
- Incremental mode: pass existing trajectories so LLM can extend vs.
  start new
- Keep the old `extract_edus_from_session` path for now during
  transition (remove in Step 5)

**Effort:** 4-6 hours

### Step 4 — Canonical keyword normalization

- On ingest: pull current project keyword cloud from SQLite
- After extraction: for each new keyword, embedding-similarity check
  against existing → auto-merge if ≥0.9, accept if <0.75, flag if between
- Log flagged cases to `~/.local/share/claude-memory/keyword_flags.jsonl`
  for periodic review

**Effort:** 2 hours

### Step 5 — Wire trajectory-mode into ingest pipeline

- Update `ingest.py` to call new trajectory extractor
- Store trajectories in SQLite, EDUs in ChromaDB with trajectory metadata
- Remove old EDU-only extraction path
- Delete+re-ingest all existing sessions (tag quality + trajectory
  assignment both require it)

**Effort:** 2-3 hours

### Step 6 — access_count tracking

- Bump `access_count` metadata on EDUs returned from retrieval
- Used by Key Decisions ranking in the index

**Effort:** 1-2 hours

### Step 7 — Index builder

- New module `src/claude_memory/index_builder.py`
- `ProjectIndex` dataclass (4 sections + keyword cloud)
- `build_index(project: str) -> ProjectIndex`
- `render_markdown(index: ProjectIndex) -> str`
- Enforce 600-token cap: truncate per-section with priority rules
- Write to `~/.local/share/claude-memory/indices/{project}.md`

**Effort:** 3-4 hours

### Step 8 — Incremental index rebuild

- After each ingest: identify touched projects, rebuild their index files
- Stable output (byte-identical when nothing changed) for prompt caching

**Effort:** 1-2 hours

### Step 9 — SessionStart hook

- `hooks/session_start.py` — reads index file for current project, emits
  `additionalContext` JSON
- Register in `~/.claude/settings.json`
- Graceful fallback: no index file → no output, no error

**Effort:** 1-2 hours

### Step 10 — Subagent retrieval MCP tool

- New MCP tool `recall_memory(search_terms: list[str], question: str)`
- Internally spawns Opus subagent via Claude CLI (`claude -p --model opus`)
- Subagent prompt instructs the strategy above
- Subagent has access to three helper tools (keyword lookup, vector
  search, window fetch) via MCP or via direct Python calls in the
  subagent's context
- Returns subagent's synthesized text to the main agent
- Replace (or deprecate) existing `search_conversation_memory` tool

**Effort:** 5-7 hours (biggest step)

### Step 11 — Benchmark

- Extend LongMemEval-Lite with "orientation" and "decision recall"
  question types
- Three conditions:
  - No memory (baseline)
  - Index only (SessionStart, no retrieval)
  - Index + subagent retrieval (full pipeline)
- Measure accuracy and latency

**Effort:** 3-4 hours

## Migration / Data Reset

Existing EDUs in ChromaDB have **no trajectory_id**. They're invisible
to the new retrieval pipeline. Before the new pipeline is functional:

**Decision: full re-ingest is required, not optional.**

Execute: `ingest_sessions(force=True)`. Cost: ~15-30 min of LLM time
(Sonnet) to re-process all known session archives with the new
trajectory-mode extractor.

Schedule this between Step 5 (wire new extractor) and Step 7 (build
index). Until re-ingest runs, the system is in a broken state — that's
fine, single-user dev box.

## Files Touched

- `src/claude_memory/extractor.py` — new trajectory-mode extraction
- `src/claude_memory/store.py` — trajectory_id/index on EDU metadata
- `src/claude_memory/trajectories.py` — **new**, SQLite trajectory store
- `src/claude_memory/keyword_canonicalizer.py` — **new**, similarity merge
- `src/claude_memory/ingest.py` — orchestrate new pipeline
- `src/claude_memory/index_builder.py` — **new**
- `src/claude_memory/server.py` — new MCP tool `recall_memory`
- `src/claude_memory/subagent.py` — **new**, Opus subagent wrapper
- `hooks/session_start.py` — **new**
- `~/.local/share/claude-memory/trajectories.db` — **new**
- `~/.local/share/claude-memory/indices/` — **new dir**
- `~/.local/share/claude-memory/keyword_flags.jsonl` — **new**

## Total Effort Estimate

~27-38 hours of focused work, roughly:
- Data model + extractor rework: ~10 hours
- Index + hook: ~6-8 hours
- Subagent retrieval: ~5-7 hours
- Migration + benchmarks: ~6-8 hours

## Dependency Order

```
Step 1 (done)
     │
     ▼
Step 2 (trajectory data model)
     │
     ▼
Step 3 (trajectory extractor) ──► Step 4 (keyword canonicalizer)
     │                                      │
     └────────────┬─────────────────────────┘
                  ▼
            Step 5 (wire into ingest)
                  │
                  ▼
           [RE-INGEST ALL SESSIONS]
                  │
                  ▼
        Step 6 ──► Step 7 ──► Step 8 ──► Step 9
     (access_ct)  (index)  (incr)  (hook)
                                          │
                                          ▼
                                    Step 10 (subagent)
                                          │
                                          ▼
                                   Step 11 (benchmark)
```

## Risks / Unknowns

- **Trajectory segmentation quality** — LLMs are variable at this. Worth
  dogfooding early with small sessions, iterate on the prompt.
- **Keyword drift** — canonical cloud quality depends on the similarity
  threshold. Start at 0.9, tune from flagged cases.
- **Subagent latency** — extra LLM call per `recall_memory` invocation.
  Worth it for quality, but main agent should know this is expensive and
  not call it every turn. The SessionStart index is the always-on
  fallback.
- **Cost** — Opus calls per memory retrieval adds up. Monitor and
  consider whether Sonnet can do the subagent role for most queries.

---

## Post-Implementation Notes

- **Re-ingest is mandatory** (not optional like in the original plan).
  Scheduled between Step 5 and Step 7.
- After Step 10 lands, consider deprecating `search_conversation_memory`
  (the old direct-query MCP tool) in favor of `recall_memory`, or keep
  both with different use cases (fast scan vs. deep recall).
