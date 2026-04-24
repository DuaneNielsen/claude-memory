# Plan: Hybrid project-boosted global memory

## Motivation

Current memory is siloed per-project by default. The recall pipeline falls back
to `project_from_cwd()`, so asking `recall_memory` in `/home/duane` only
searches the `home` project even though the actual content (e.g. "hallucination
guardrails for claude-memory") lives in `projects-claude-memory`.

User insight: cross-project knowledge transfer matters. The current project
should be a *contextual signal* (boost) — not a *hard filter*.

Goal: memory is global by default; current project skews ranking but never
excludes cross-project hits.

## Design

Project filter becomes a **boost signal** in ranking, not a hard filter.

Scoring:
```
score = similarity × recency_weight × project_multiplier
project_multiplier = PROJECT_BOOST_FACTOR if result.project == current_project else 1.0
```

Default `PROJECT_BOOST_FACTOR = 1.5` (empirically tune). Explicit hard filter
remains available for when the user wants project-isolated search.

## Current state (what to change)

| Component | Current behavior | New behavior |
|---|---|---|
| `search_conversation_memory` (MCP) | Optional `project` → hard filter in ChromaDB `where` clause | `project` → boost factor; add separate `strict_project` arg for hard filter |
| `recall_memory` (MCP) | Falls back to `project_from_cwd()` as hard filter | Current project → boost in ranking; global search is default |
| `claude-memory recall` (CLI) | `--project` → hard filter | `--project` → boost; add `--strict-project` flag |
| `claude-memory search` (CLI) | `--project` → hard filter | Same change as above |
| SessionStart hook | Injects current project's index only | *No change* — project index gives local context, recall is global |

## Implementation steps

### 1. Config
- Add `PROJECT_BOOST_FACTOR = 1.5` to `src/claude_memory/config.py`
- Tunable without code change later

### 2. Store layer (`src/claude_memory/store.py`)
- Drop `where={"project": project}` from vector-search queries when boost mode
- Return all candidates with project metadata attached
- Keep `where` path for `strict_project` mode

### 3. Query layer (`src/claude_memory/query.py`)
- `search()` signature: `search(store, query, current_project=None, strict_project=None, max_results)`
  - `current_project` → boost signal
  - `strict_project` → hard filter (for explicit scoped queries)
- After vector search, apply boost: `r.score *= PROJECT_BOOST_FACTOR if r.project == current_project`
- Re-sort by boosted score, return top N

### 4. Retrieval layer (`src/claude_memory/retrieval.py`)
- `_find_hit_trajectory_ids`: remove hard project filter from keyword + vector queries
- Apply boost during scoring
- `recall_memory()` signature: rename `project` → `current_project` semantically; add `strict_project`
- Default: no strict filter, use current project as boost only

### 5. Trajectory store (`src/claude_memory/trajectories.py`)
- Add `search_by_keywords_global(keywords) -> list[(Trajectory, score)]` that spans all projects
- Keep existing `search_by_keywords(keywords, project)` for strict mode
- Similar changes for any other project-filtered helpers

### 6. MCP tool (`src/claude_memory/server.py`)
- Update `recall_memory` tool schema: rename/clarify `project` arg, add `strict_project`
- Update `search_conversation_memory` tool schema: same
- Tool descriptions: explain boost semantics so the calling agent picks the right one

### 7. CLI (`src/claude_memory/cli.py`)
- `search` / `recall` subcommands: `--project` becomes boost, add `--strict-project` for hard filter

### 8. Tests / validation
- Run `claude-memory recall "hallucination guardrails"` from `/home/duane` — should surface
  `projects-claude-memory` results with `home` results visible too
- Run with `--strict-project home` — should return only `home` results (or none)
- Run with explicit `--project projects-claude-memory` — should boost that project

## Open questions

- **Boost factor calibration**: 1.5 is a guess. 2.0 may still let global dominate;
  3.0 may over-suppress cross-project. Needs empirical comparison. Parameterize and tune.
- **Session-level boost**: should same-session trajectories get an even higher boost
  than same-project ones? Probably yes — within the same conversation, context is
  extremely relevant. `SESSION_BOOST_FACTOR = 2.0` maybe.
- **SessionStart hook enrichment**: should we also inject a "recent cross-project
  activity" section (top N items from OTHER projects)? Adds context at ~200 extra
  tokens. Could be opt-in.
- **Recency + boost interaction**: currently multiplicative. Alternative: additive
  or weighted average. Multiplicative is simpler; revisit if results feel wrong.
- **strict_project semantics**: when should the tool default to strict vs. boost?
  Probably always boost default, strict only when explicitly requested ("search
  only my frigate stuff"). Tool description should steer the calling agent.

## Rollout

- Implement in sequence: config → store → query → retrieval → server/CLI
- Test each layer in isolation where possible
- Final end-to-end test: the "hallucination guardrails" cross-project recall
- Commit as a single changeset (small enough to reason about together)
- No data migration needed — all existing state remains valid
