# Plan: Subagent Retrieval Filter

Add an LLM-based relevance filter between dense retrieval and final result
return. Based on EMem's finding that the LLM filter is the single biggest
performance gain in their ablation — especially for multi-hop queries.

## Goal

Given a query and ~50-100 candidate EDUs from dense retrieval, use an LLM to
produce a curated set of ~5-10 relevant EDUs to return. Beat the current
`similarity × recency` ranking on Recall@10 and LLM-judge accuracy.

## Design

### Architecture choice: inline function first, subagent second

**Phase A** — implement as inline function called by `query.py`. Easier to
iterate on the prompt, easier to benchmark, no MCP/subagent registration
complexity. Expose as a boolean config flag (`FILTER_ENABLED`).

**Phase B** — if filter shows significant benchmark gain (>5pp on LongMemEval-S),
register as a Claude Code subagent_type so the main agent can delegate more
complex memory work to it (read full session transcripts, do multi-step
reasoning, return synthesized context).

### Model choice

- **Primary: Haiku (claude-haiku-4-5-20251001)** — fast (~500-800ms), cheap
  ($0.001-0.002/query at filter token sizes), adequate for relevance judgment.
- **Fallback: local Qwen3.5-27B via llama-server** — eliminates network
  dependency, uses existing infrastructure. Slower (1-2s), no per-query cost.
- **Config flag to switch**: `FILTER_MODEL = "haiku" | "local"`.

### Input / output

**Input to filter:**
- User's query string
- Top-N candidates from dense retrieval (N configurable, default 50)
- Each candidate: EDU text, timestamp, project, session ID, similarity score

**Output from filter:**
- List of EDU IDs to keep (ordered by relevance)
- Optional: confidence score per EDU
- Optional: one-line justification per EDU (for debugging)

**Not synthesized**: the filter picks EDUs, it doesn't rewrite them. This
matches EMem. Synthesis happens in the main agent's response if needed.

### Filter prompt structure

Based on EMem's "maximally inclusive" philosophy. Key instructions:

1. Be maximally inclusive — false positives are cheap, false negatives lose
   information
2. Return IDs for EDUs that *could* be relevant, even if uncertain
3. Order by relevance
4. If fewer than K candidates are truly relevant, return fewer (don't pad)
5. If nothing is relevant, return empty list (don't guess)

### Prompt draft

```
You are a memory retrieval filter. Given a query and candidate memories,
select the subset that could help answer the query.

Query: {query}

Candidates:
[{id}] {text} (project: {project}, {days_ago} days ago)
[{id}] {text} ...

Rules:
- Be maximally inclusive — prefer false positives over missing relevant info
- Return IDs ordered by relevance (most relevant first)
- If fewer candidates are relevant than {max_k}, return fewer
- If nothing is relevant, return []
- Output format: JSON list of IDs, e.g. ["edu-123", "edu-456"]
```

## Implementation Steps

### Step 1 — Baseline benchmark (required, before any changes)

Before implementing, establish current numbers to beat:
- Run `benchmarks/longmemeval_lite.py` on current system (similarity × recency)
- Record per-category accuracy + overall
- Record Recall@10

**Deliverable**: `benchmarks/results/baseline_pre_filter.json`

### Step 2 — Filter function skeleton

Add `filter.py` module:

```python
@dataclass
class FilterResult:
    kept_ids: list[str]
    confidence: dict[str, float]  # optional
    latency_ms: float

def llm_filter(
    query: str,
    candidates: list[SearchResult],
    max_k: int = 10,
    model: str = "haiku",
) -> FilterResult: ...
```

Integration point: `query.py::search()` adds optional filter stage after
similarity×recency scoring, before `results[:max_results]` cut.

### Step 3 — Claude API client

Use `anthropic` Python SDK with prompt caching enabled. Cache the system
prompt + rules section (stable). Only the candidates vary per call.

Config additions to `config.py`:
- `FILTER_ENABLED: bool = False` (default off during development)
- `FILTER_MODEL: str = "haiku"`
- `FILTER_CANDIDATES: int = 50` (candidates to feed filter; larger than current 30)
- `FILTER_MAX_K: int = 10` (max to keep after filter)

### Step 4 — Prompt iteration

Hand-test on 10-20 known queries from the session archive. Iterate on prompt
until the filter:
- Correctly prunes obviously-irrelevant candidates (topic mismatch)
- Keeps candidates that are tangentially relevant
- Handles "nothing relevant" case without hallucinating

Track: which kinds of queries confuse the filter? Temporal? Multi-entity?

### Step 5 — Benchmark A/B

Run LongMemEval-S with `FILTER_ENABLED=false` vs `true`. Compare:
- Overall accuracy
- Per-category accuracy (multi-session expected to gain most)
- Latency (expect +500-1500ms per query)
- Cost ($ per 1000 queries)

**Pass criteria**: filter must produce ≥3pp overall accuracy gain, or ≥5pp
on multi-session reasoning category. Otherwise it's not worth the cost.

### Step 6 — Candidate pool sweep

Filter's value grows with candidate pool size (EMem philosophy: cast wide,
let the LLM prune). Benchmark `FILTER_CANDIDATES ∈ {20, 50, 100, 150}`.

Find the knee: where does more candidates stop helping? Cost scales linearly
with pool size, quality flattens.

### Step 7 — Caching

Two caching opportunities:

1. **Prompt caching** (Anthropic API) — stable system prompt block, variable
   candidates. Already supported by the SDK. Near-free after first call.

2. **Result caching** (local) — if the same query + candidate set comes up
   (rare but possible), cache the filter output keyed on
   `hash(query + sorted_ids)`. SQLite table with TTL.

### Step 8 — Subagent registration (Phase B, conditional)

**Only proceed if Step 5 showed significant benefit.**

Register as Claude Code subagent:

```yaml
---
name: memory
description: >
  Expert on past Claude Code conversations. Use this agent when the user
  references prior work, asks "did we...", or when historical context would
  improve the current response. Can search memory, read full session
  transcripts, and synthesize findings.
tools: [search_conversation_memory, Read, Grep]
---
```

Subagent gets:
- Its own context (doesn't pollute main)
- Ability to do multiple searches + session file reads
- Returns synthesized answer, not raw EDU dump

Benefit over inline filter: multi-step reasoning ("first search X, then check
if the answer references session Y, read that session, now I have full
context").

## Testing

### Unit tests
- Filter with 10 obvious candidates + 1 obvious answer → returns the 1
- Filter with no relevant candidates → returns []
- Filter with LLM JSON parse failure → falls back to similarity-only ranking
- Filter with network error → falls back gracefully, logs warning

### Integration tests
- End-to-end: ingest known sessions, query for known facts, verify filter
  returns the right EDUs

### Benchmark tests
- LongMemEval-S runs with and without filter
- Regression test: filter never *decreases* Recall@K vs no-filter

## Success Metrics

- **Primary**: ≥3pp accuracy gain on LongMemEval-S overall
- **Primary**: ≥5pp gain on multi-session reasoning category
- **Secondary**: Latency <1500ms p95 with filter enabled
- **Secondary**: Cost <$0.002/query at 50 candidates

If primary metrics miss: don't ship the filter. Either the prompt needs work
or the concept doesn't apply to this corpus.

## Risks / Unknowns

- **Corpus difference**: EMem's filter was validated on LoCoMo (persona chat).
  Our corpus is technical. Filter may behave differently — some facts only
  make sense in technical context the filter LLM lacks.
- **Latency sensitivity**: +1s per query may be noticeable. User can disable
  via config if it hurts UX.
- **Prompt tuning time**: could take 5-10 iterations to get the prompt right.
  Budget for this.

## Files Touched

- `src/claude_memory/filter.py` (new)
- `src/claude_memory/query.py` (integration)
- `src/claude_memory/config.py` (new flags)
- `pyproject.toml` (add `anthropic` dep)
- `benchmarks/results/` (new dir for tracked results)
- `.claude/agents/memory.md` (Phase B only)

## Estimated Effort

- Phase A (inline filter + benchmark): **4-6 hours** of focused work
- Phase B (subagent registration): **2-3 hours** (only if Phase A wins)
