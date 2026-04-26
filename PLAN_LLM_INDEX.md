# PLAN — LLM-curated project index

**Status:** drafted 2026-04-26, not yet implemented. Delete this file after the feature ships and the design is reflected in `ARCHITECTURE.md`.

## Goal

**The index is a catalog, not a digest.** Its job is to advertise to Claude what memories are *available* in the store so he knows what to ask for via `recall_get_context`. The actual content of those memories lives in ChromaDB and is retrieved on demand — it does not need to be pre-injected at SessionStart.

Replace the deterministic `build_index()` projection with a one-LLM-call curation step. Pipeline pulls a **wide** slice of the project's distilled data (trajectory summaries, decision/gotcha/preference EDUs, keyword frequencies), feeds it to the strongest available model (Opus), and gets back a structured catalog: a project overview, behavior-shaping preferences (these *do* need to be content because Claude can't act on them via a tool call), and inventories that group available memories by topic — counts and keyword groupings, not full EDU text.

The current builder has three problems:
1. It dumps 8 full-text decision/gotcha EDUs into the system prompt that mostly aren't relevant to the upcoming conversation. Wasted tokens, and worse: it primes Claude on facts that may be superseded, instead of letting him decide what to fetch.
2. It can't write a project overview or detect active threads.
3. It can't suppress entries that have been superseded ("EasyEffects removed" appears in Recent Activity even though it's now in CLAUDE.md and irrelevant).

An LLM pass fixes (2) and (3). The catalog reframing fixes (1).

### Content vs reference (the cut)

| Section | Type | Why |
|---|---|---|
| `project_overview` | content (~2 sentences) | Claude needs to orient before any tool call; can't be deferred. |
| `preferences` | content (full EDU text) | These shape behavior — Claude must obey them before he could think to call a tool. Cannot be deferred. |
| `available_memories` | **reference** (counts + topic groupings) | "23 decisions tagged: pipewire, claude-memory, niri, …" — Claude sees what exists and calls `recall_get_context` to pull what's relevant. |
| `active_threads` | reference (one-line summaries) | These are pointers to ongoing trajectories. Useful at orient time; full content via recall. |
| `recent_activity` | reference (one-line trajectory summaries) | Same — already a catalog, no full text. |
| `keyword_cloud` | reference (keywords only) | **Always present.** Filtered by frequency if too large. Doubles as "here are the topics this project's memory covers." |

## Design decisions (locked unless flagged 🟡)

### 1. Input to the LLM (the "wide query")

The LLM needs **wide enough input to group EDUs by topic**, but does NOT need full-text dumps for tags whose output will be a count+topics catalog. So we vary detail by tag:

Per touched-project, gather:
- **Project metadata**: name, total trajectory count, total EDU count, time range covered.
- **Trajectory digest**: all trajectories, each as `{id, date, summary, keywords, edu_count}`. Sort newest-first. Cap at 80 (subsample by recency × max-keyword-freq if more).
- **Preference EDUs (verbatim)**: ALL preference-tagged EDUs (cap 30). Full text — these may be output verbatim.
- **Catalog-only EDUs (text + tag + keywords)**: for tags `decision`, `gotcha`, `architecture`, `config`, `project`:
  - Include `{edu_id, tag, text, keywords-from-trajectory, date}` so the LLM can group by topic.
  - Cap per tag: 80 most-recent EDUs. (Higher than the digest design's 50 since we're paying for grouping accuracy, not for the LLM to memorize text.)
  - Old EDUs that don't fit the cap still count toward `available_memories[tag].count` — pass the count separately.
- **Keyword cloud with frequencies**: full canonical keyword list + occurrence counts.
- **Per-tag full counts**: `{decision: 47, gotcha: 23, …}` so `available_memories[].count` is accurate even when EDUs are subsampled.

Estimated input: 6–12k tokens for a typical project, capped via subsampling.

🟡 **Open**: include `config` and `project` tagged EDUs in the catalog? They tend to be high-volume noise — a "247 config entries" inventory line might be useless. Could exclude or merge into a single "other" tag. Resolve by inspecting an actual prompt input dump in the first build session.

### 2. Model

`claude-opus-4-7` (or whichever is current) via the existing `claude -p` CLI in `extractor.call_claude`. The CLI already supports `--model`, `--system-prompt`, `--json-schema`, `--output-format json` — we reuse all of that infrastructure.

### 3. Output schema

```json
{
  "project_overview": "1-2 sentences on what this project is and the current focus area.",
  "preferences": [
    {"text": "...", "source_edu_ids": ["..."]}
  ],
  "available_memories": [
    {
      "tag": "decision",
      "count": 47,
      "topics": ["pipewire", "claude-memory", "niri", "build-from-source"]
    },
    {
      "tag": "gotcha",
      "count": 23,
      "topics": ["pipewire", "niri", "audio"]
    },
    {
      "tag": "architecture",
      "count": 31,
      "topics": ["claude-memory", "ingestion", "trajectories"]
    }
  ],
  "active_threads": [
    {"summary": "currently mid-...", "keywords": ["..."]}
  ],
  "recent_activity": [
    {"date": "2026-04-26", "summary": "..."}
  ],
  "keyword_cloud": ["..."]
}
```

Notes:
- `available_memories` is the catalog/inventory — counts and topic groupings per tag. Claude reads this and calls `recall_get_context` with the relevant `tag` + `topics` if he needs the actual content. **This replaces the old full-text `key_decisions` / `key_gotchas` sections.**
- `project_overview` and `active_threads` are **new** sections — the deterministic builder can't produce them.
- `preferences` is the only section that still includes full EDU text (because preferences shape behavior immediately and can't be deferred to a tool call).
- `source_edu_ids` is required on `preferences` for traceability. The LLM must select from the EDUs we showed it; enforced by prompt + schema.
- Caps: 8 preferences, 6 active threads, 10 recent activity, 40 keywords. `available_memories` has one entry per tag (closed set: 6 entries max). No cap on `topics` per tag at this layer — `enforce_token_budget` trims if needed.

### 3a. Rendered markdown

The wrapper text already injected by the SessionStart hook will be expanded to make the catalog framing explicit:

```
# Conversation memory index — project `<name>`

The sections below catalog what memories are available for this project.
Most are pointers — call `recall_get_context` (via an Agent/Task subagent)
to fetch the actual content for any topic of interest.

The "Preferences" section is the exception: those shape behavior and apply
immediately, so the full text is included.

---
```

Then the body, e.g.:
```
## Project overview
claude-memory is a local conversation-memory plugin for Claude Code.
Currently focused on improving the SessionStart index quality (LLM-curated catalog).

## Preferences
- For 'why did you decide X' questions, dispatch recall_get_context via an Agent.
- A dedicated CI API key (not the dev's personal key) should be used for CI.

## Memories available (call recall_get_context to retrieve)
- **47 decisions** across topics: pipewire, claude-memory, niri, build-from-source, …
- **23 gotchas** across topics: pipewire, niri, audio, …
- **31 architecture notes** across topics: claude-memory, ingestion, trajectories, …
- **18 config entries** across topics: pipewire, niri, transcriber, …
- **12 project notes** across topics: claude-memory, frigate, …

## Active threads
- LLM-curated index design (claude-memory, index-builder, planning)
- CI deploy-matrix expansion (claude-memory, ci, docker)

## Recent activity
- [2026-04-26] Built and validated the claude-memory CI pipeline
- [2026-04-26] Designed a four-layer GitHub Actions CI pipeline for claude-memory
- …

## Keyword cloud
agent-framework, ci, claude-code, claude-memory, chromadb, …
```

Notice how dramatically smaller this is than the old design's "full text of 8 decisions + 8 gotchas". The whole body for an active project should fit in ~300-400 tokens.

### 4. Prompt sketch

System prompt should mirror the existing extractor prompts in shape and explicitly state the catalog framing:

- **Lead with intent**: "You are building a CATALOG of memories available for a project, not a digest of their content. Most output is pointers (counts + topic keywords) so a downstream agent can decide what to fetch via `recall_get_context`. Full content is included ONLY for the `preferences` section because preferences shape behavior immediately."
- **CRITICAL SOURCE BOUNDARY**: only synthesize from the EDUs and trajectory summaries provided; do not pull from CLAUDE.md, training, or surrounding files.
- **Section-by-section instructions**:
  - `project_overview`: 1-2 sentences. What is this project, and what's the current focus? Inferred from the trajectory summaries' temporal pattern.
  - `preferences`: select up to 8 preference-tagged EDUs that still apply. Drop ones superseded by later EDUs. Keep verbatim text.
  - `available_memories`: for each tag in {decision, gotcha, architecture, config, project}, group the EDUs of that tag into 3-7 topic clusters. Each cluster = a short keyword (matching the project's canonical keyword cloud where possible). Output the count of EDUs and the topic-cluster keywords. Do NOT include EDU text. Do NOT manufacture topics — every topic keyword must appear in the project's keyword cloud or be a clear merge of two existing keywords.
  - `active_threads`: trajectories where the most recent EDU suggests the work is still in motion (open questions, "to do", mid-decision phrasings). Up to 6. One-line summary + keywords.
  - `recent_activity`: top 10 trajectories by recency. One-line summaries (you can use the trajectory's existing summary verbatim or rewrite for clarity).
  - `keyword_cloud`: the canonical project keywords, alphabetized. Always present even if other sections are empty. Filter to top-40 by frequency if the cloud is larger.
- **Quality bar**: "Prefer specific over general. Drop entries superseded by later EDUs. Topic keywords in `available_memories` should be the same vocabulary the user uses elsewhere — match the project's keyword cloud."
- **One-shot example**: worked input/output using `home` or `projects-claude-memory` as the example (whichever has the richest data when we draft).

Draft v1, run on 3 real projects, eyeball output, iterate prompt + one-shot until quality is consistently good.

### 5. Render path

The LLM returns structured JSON; `render_markdown()` is rewritten to render it. No more "empty section omitted" branching — the LLM has already decided what's worth including.

The wrapped output (header + content) injected at SessionStart stays the same. Only the body changes.

### 6. Determinism / cache

The LLM output is **not byte-deterministic**, so prompt-cache hits across SessionStart in different sessions are weakened (the wrapper text is still cache-stable, but the body changes whenever the index rebuilds).

Mitigation: **content-hash gating in `write_index`**. Compute a hash of the LLM input (sorted trajectory IDs + EDU IDs + their text hashes). If hash is unchanged since the last index for this project, skip the LLM call and reuse the existing index file. Hash gets stored alongside the index as `<project>.md.meta.json`.

This makes the LLM call fire only when something genuinely new arrived — typically once per ingest run, sometimes zero (re-ingest of unchanged session).

### 7. Cost budget

The catalog framing dramatically shrinks the OUTPUT (no full-text EDU dumps). Input stays wide — the LLM still needs to see all the EDUs to group them by topic intelligently.

- Input: ~6–12k tokens per project per ingest run when there's new content
- Output: ~250–400 tokens (was ~600 in the digest design, since most sections are now references not content)
- Per-call cost: rough estimate $0.05–0.15 at current Opus pricing
- Per-day cost: typical ingest touches 1–3 projects → $0.05–$0.45/day for active users

If this turns out wrong by 10× we can: (a) downgrade to Sonnet for index curation; (b) reduce the input size cap; (c) re-introduce the deterministic builder as a "no-key" fallback.

### 7a. Token budget at SessionStart

Even more important than $ cost: the index gets injected into every SessionStart in this project, so its size is paid in tokens on every conversation start. The catalog design should land at ~200-400 tokens for typical projects (down from the deterministic builder's 600 budget). Keep `MAX_INDEX_TOKENS = 600` as the hard cap; expect to be well under it.

### 8. Failure modes & fallbacks

- **LLM rate-limited**: catch `RateLimitError` from `call_claude`; fall back to the deterministic builder for this run; log a warning.
- **Malformed JSON output**: catch `json.JSONDecodeError`; same fallback.
- **LLM produces a `source_edu_ids` not in the input**: log warning; drop the offending entry; render the rest.
- **`claude` CLI missing**: the deterministic builder runs unconditionally as a fallback so the system still works without API credentials (important for users running just `claude-memory ingest` on parsed data).

The deterministic builder stays in place as `_build_index_deterministic()`. The new path is `_build_index_llm()`. `build_index()` becomes a thin chooser: try LLM, fall back to deterministic.

## Implementation steps

Ordered. Each step should be a single commit; cumulative diff stays small.

### Step 1 — refactor existing builder for fallback path

`src/claude_memory/index_builder.py`:
- Rename current `build_index` → `_build_index_deterministic`. No behavior change.
- Add `def build_index(...)` thin wrapper that just delegates. **One commit, pure rename + delegation.**

### Step 2 — input gathering

Add `_gather_index_input(project, traj_store, mem_store)` returning a typed dataclass `IndexInput` with all the fields from §1. Pure data assembly, no LLM, no rendering. Easy to unit-test by snapshot.

### Step 3 — output schema + prompt

Add `INDEX_CURATOR_SYSTEM_PROMPT`, `INDEX_CURATOR_JSON_SCHEMA`, `INDEX_CURATOR_ONE_SHOT_INPUT/OUTPUT` constants alongside the others in `extractor.py`. Mirrors the existing `LABEL_SYSTEM_PROMPT` style for consistency.

The one-shot can be hand-written from a real `IndexInput` dump for `home` or `projects-claude-memory`.

### Step 4 — LLM call wrapper

Add `async def curate_index_with_llm(input: IndexInput, model: str | None = None) -> CuratedIndex` in `index_builder.py` that calls `extractor.call_claude` with the new prompt + schema. Returns a structured object, not a markdown string.

### Step 5 — Markdown renderer for curated index

`render_curated_markdown(curated: CuratedIndex) -> str`. Drops empty sections. Writes the wrapper-friendly form (no leading h1 — that comes from the SessionStart hook).

### Step 6 — Wire up `build_index`

`build_index` now:
1. Gathers `IndexInput`
2. Computes input hash
3. If hash matches stored `<project>.md.meta.json`, return early (reuse existing index)
4. Otherwise: attempt `curate_index_with_llm`. On any exception, fall back to `_build_index_deterministic`.
5. Render and write atomically. Update `<project>.md.meta.json` with the new hash.

### Step 7 — Token budget enforcement

Keep `enforce_token_budget()` but adapt it to the new structure (sections in priority order: project_overview > preferences > active_threads > key_decisions > key_gotchas > recent_activity > keyword_cloud). The LLM should respect the budget but we still trim defensively.

### Step 8 — Smoke test in CI

Add `tests/smoke/test_index_curator.py` that:
- Builds a synthetic `IndexInput` with a handful of EDUs / trajectories
- Calls `render_curated_markdown` on a fake `CuratedIndex` and asserts shape
- Does NOT call the LLM (Layer 3 is keyless — the LLM call lives in the future end-to-end test workflow)

Add a separate **keyed** workflow file `.github/workflows/index-curator.yml` (gated on `workflow_dispatch` only at first) that runs `claude-memory reindex --project projects-claude-memory` and posts the resulting index to the run summary so we can eyeball quality without running locally.

### Step 9 — Manual quality pass

Before merging:
- Run `claude-memory reindex` for at least 3 projects with varied profiles (active, dormant, mixed-tag)
- Eyeball each rendered index — does it read like a useful one-page brief?
- Iterate prompt + one-shot until 3 out of 3 read well

### Step 10 — Update SessionStart hook wrapper text

`hooks/session_start_index.py` currently wraps the index body with a 5-line generic header. Replace with the catalog-framing wrapper sketched in §3a — the existing wording understates the "this is a catalog, fetch via recall" framing and Claude needs to hear it explicitly to use the index correctly.

### Step 11 — Update docs

- `README.md`: update the "How it works" section's mermaid to show the LLM index step. Update the "What it adds" section to reflect that the SessionStart hook now injects a catalog rather than a digest.
- `ARCHITECTURE.md`: section on the index builder, with the catalog vs digest framing explained.
- `PROMPTS.md`: add the new `INDEX_CURATOR_SYSTEM_PROMPT` to Part 1.
- Delete this `PLAN_LLM_INDEX.md`.

## Out of scope (for now)

- **Cross-project synthesis** (an "all projects" overview) — could be useful but doubles cost. Defer.
- **Active-thread detection** automation — for now the LLM infers it from recency + open questions in EDU text. Could later be supplemented with an explicit "open" flag at extraction time.
- **Streaming index updates** — currently the index rebuilds atomically end-of-ingest. No streaming partial updates.
- **A/B comparing LLM vs deterministic on retrieval quality** — interesting research question, but the goal here is index *quality for human + LLM consumption at SessionStart*, not retrieval. Different metric.

## Things to decide WITH the user when the new context starts

1. **🟡 Include `config` / `project` tagged EDUs in the input?** Try without first; revisit if the project_overview looks impoverished.
2. **🟡 Should the meta file be `.meta.json` or embedded as YAML frontmatter in the `.md`?** Frontmatter is more elegant; separate file is simpler. Lean toward separate file for now.
3. **🟡 Cost cap** — should ingest abort the LLM curation step if estimated input > 30k tokens? Or just truncate aggressively?
4. **🟡 Keep the deterministic fallback forever, or rip it out after a few weeks of dogfooding?** Lean toward keeping it indefinitely since it costs nothing to maintain and gives offline / no-API users a working system.

## Pre-flight checklist for the build session

- [ ] Confirm Opus model ID currently exposed by the `claude` CLI (`claude --model claude-opus-4-7` works?)
- [ ] Decide on prompt input cap (default proposed: 80 trajectories, 50 decisions, 50 gotchas, 30 preferences, 20 architecture)
- [ ] Stand up a tiny throwaway script that dumps `IndexInput` for `projects-claude-memory` to a file so we can paste it into the one-shot draft
- [ ] Read this plan top-to-bottom once and surface any disagreement before writing code
