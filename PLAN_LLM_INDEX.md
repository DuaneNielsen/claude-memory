# PLAN — LLM-curated project index

**Status:** drafted 2026-04-26, not yet implemented. Delete this file after the feature ships and the design is reflected in `ARCHITECTURE.md`.

## Goal

Replace the deterministic `build_index()` projection with a one-LLM-call distillation step. Pipeline pulls a **wide** slice of the project's distilled data (trajectory summaries, decision/gotcha/preference EDUs, keyword frequencies), feeds it to the strongest available model (Opus), and gets back a curated `ProjectIndex` shaped for SessionStart injection. Acceptable cost trade for materially higher index quality.

The current builder is a rank-and-truncate over already-extracted facts. It can't synthesize across trajectories ("the user has been moving from FastAPI to Litestar over the last 3 weeks"), can't write a project overview, and can't suppress entries that have been superseded ("EasyEffects removed" appears in Recent Activity even though it's now in CLAUDE.md and irrelevant). An LLM pass fixes all three.

## Design decisions (locked unless flagged 🟡)

### 1. Input to the LLM (the "wide query")

Per touched-project, gather:
- **Project metadata**: name, total trajectory count, total EDU count, time range covered.
- **Trajectory digest**: all trajectories for the project, each as `{id, date, summary, keywords, edu_count}`. Sort newest-first. Cap at 80 (subsample by recency × max-keyword-freq if more).
- **Selected EDUs verbatim** (full text, with tag and date):
  - All `preference` EDUs (cap 30)
  - All `decision` EDUs from the last 90 days (cap 50)
  - All `gotcha` EDUs from the last 90 days (cap 50)
  - Top 20 `architecture` EDUs by recency × keyword-freq
- **Keyword cloud with frequencies**: full canonical keyword list + occurrence counts.

Estimated input: 6–15k tokens for a typical project, capped via subsampling.

🟡 **Open**: should we include a sample of `config` and `project` tagged EDUs? Probably not — they're high-volume noise. Resolve in the first build session by inspecting an actual prompt input dump.

### 2. Model

`claude-opus-4-7` (or whichever is current) via the existing `claude -p` CLI in `extractor.call_claude`. The CLI already supports `--model`, `--system-prompt`, `--json-schema`, `--output-format json` — we reuse all of that infrastructure.

### 3. Output schema

```json
{
  "project_overview": "1-2 sentences on what this project is and the current focus area.",
  "preferences": [
    {"text": "...", "source_edu_ids": ["..."]}
  ],
  "key_decisions": [
    {"text": "...", "rationale": "why X over Y", "date": "2026-04-22", "source_edu_ids": ["..."]}
  ],
  "key_gotchas": [
    {"text": "...", "date": "2026-04-15", "source_edu_ids": ["..."]}
  ],
  "active_threads": [
    {"summary": "currently mid-...", "last_activity": "2026-04-25", "next_step": "..."}
  ],
  "recent_activity": [
    {"date": "2026-04-26", "summary": "..."}
  ],
  "keyword_cloud": ["..."]
}
```

Notes:
- `project_overview` and `active_threads` are **new** sections — the deterministic builder can't produce them.
- `key_decisions` and `key_gotchas` are split (currently they share a single 8-slot section that gotchas dominate).
- `source_edu_ids` is required for traceability — we want to be able to say "this index claim came from EDUs X, Y, Z" for debugging/auditing. The LLM must select from the EDUs we showed it; this is enforced by the prompt + schema.
- Caps in the prompt: 8 preferences, 8 decisions, 5 gotchas, 4 active threads, 8 recent activity, 40 keywords.

### 4. Prompt sketch

System prompt should mirror the existing extractor prompts in shape:
- "CRITICAL SOURCE BOUNDARY" clause: only synthesize from the EDUs and trajectory summaries provided in the user message; do not pull from CLAUDE.md, training, or surrounding files.
- Section-by-section instructions for what each output field should be (project_overview = "what would a new collaborator need to know in one breath", active_threads = "things still mid-flight, not settled facts", etc.).
- Quality bar: "prefer specific over general", "name files / functions / paths when relevant", "drop entries that have been superseded by later EDUs", "phrasings should be skimmable in <2s each".
- One-shot example with worked input/output (using the home project as the example, since it has rich data).

Draft a first version, run it on `home` and `projects-claude-memory`, iterate until the output reads well to a human.

### 5. Render path

The LLM returns structured JSON; `render_markdown()` is rewritten to render it. No more "empty section omitted" branching — the LLM has already decided what's worth including.

The wrapped output (header + content) injected at SessionStart stays the same. Only the body changes.

### 6. Determinism / cache

The LLM output is **not byte-deterministic**, so prompt-cache hits across SessionStart in different sessions are weakened (the wrapper text is still cache-stable, but the body changes whenever the index rebuilds).

Mitigation: **content-hash gating in `write_index`**. Compute a hash of the LLM input (sorted trajectory IDs + EDU IDs + their text hashes). If hash is unchanged since the last index for this project, skip the LLM call and reuse the existing index file. Hash gets stored alongside the index as `<project>.md.meta.json`.

This makes the LLM call fire only when something genuinely new arrived — typically once per ingest run, sometimes zero (re-ingest of unchanged session).

### 7. Cost budget

- Input: ~10k tokens per project per ingest run when there's new content
- Output: ~600 tokens
- Per-call cost: rough estimate $0.10–0.20 at current Opus pricing
- Per-day cost: depends on how many distinct projects get touched per ingest run. Typical ingest touches 1–3 projects → $0.10–$0.60/day for active users.

If this turns out wrong by 10× we can: (a) downgrade to Sonnet for index curation; (b) reduce the input size cap; (c) re-introduce the deterministic builder as a "no-key" fallback.

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

### Step 10 — Update docs

- `README.md`: update the "How it works" section's mermaid to show the LLM index step.
- `ARCHITECTURE.md`: section on the index builder.
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
