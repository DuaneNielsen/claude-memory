# Plan: Replace `claude -p` subagent with caller-driven Agent dispatch

## Motivation

`recall_memory` currently spawns `claude -p` as a synthesis subagent. Phase
timings on a typical recall (logged in `~/.local/share/claude-memory/recall_log.jsonl`):

```
total ≈ 21s
  find        3s   (vector search; chromadb hiccups push to 13s sometimes)
  gather    0.05s
  render    0.001s
  subagent   17–20s
    boot          1.5–7.5s   ← variance from SessionStart hooks + MCP fanout
    api_req       1–6s       ← API TTFT, scales with model
    gen           14–28s     ← actual model work
    cleanup       0.4s
```

The `subagent.boot` phase is pure overhead with nothing to show for it: claude
CLI cold-start, SessionStart hooks running for *every* recall, MCP servers
re-connecting, plugin sync. None of it helps synthesize an answer. It varies
1.5s–7.5s with no signal — just noise polluting the latency.

Caller-driven Agent dispatch sidesteps it entirely. Claude Code's `Agent` /
`Task` tool already does in-process subagent dispatch with its own context
window. No CLI cold start. No hook fanout. No plugin sync. Same OAuth as the
main session.

## Design

Split `recall_memory` into two responsibilities:

1. **MCP tool**: returns the stitched wall-of-text. No synthesis.
2. **Caller** (the main agent in Claude Code): dispatches an Agent-tool
   subagent with a prompt that tells it to call the MCP tool, read the wall,
   and synthesize.

```
┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Main Claude      │     │ Agent subagent   │     │ MCP server         │
│                  │ →→  │ (in-process)     │ →→  │ recall_get_context │
│ "user asked X"   │     │  prompt: synthe- │     │  → wall-of-text    │
│                  │     │  size from wall  │     │                    │
└──────────────────┘     └──────────────────┘     └────────────────────┘
        ↑                       ↓
        └──── synthesized answer ────┘
```

The Agent subagent is the same context-isolation primitive `claude -p` was
giving us, just without the boot tax. The wall lives in the subagent's
context, never the main agent's.

## Current state (what to change)

| Component | Current | New |
|---|---|---|
| MCP `recall_memory` | builds wall, spawns `claude -p`, returns synthesized prose | **deleted** (or kept briefly as a deprecated fallback) |
| MCP `recall_get_context` | does not exist | new tool: returns wall-of-text + diagnostics |
| `call_subagent` in `retrieval.py` | spawns `claude -p` | **deleted** |
| `recall_memory()` Python fn | end-to-end including subagent | renamed/refactored: returns `(wall, diagnostics)` |
| Tool description for `recall_get_context` | n/a | explicit: "always dispatch via Agent tool; never call directly from main context" |
| `RecallResult` dataclass | carries answer + subagent_breakdown | drop subagent fields; keep find/gather/render/total |
| Recall log JSONL | logs subagent timings | drops subagent timings; the calling agent's Agent-tool dispatch is now the synthesis cost and isn't ours to measure |

## Implementation steps

### 1. New MCP tool: `recall_get_context`

`server.py`:
- Replace the `recall_memory` Tool entry with `recall_get_context`.
- Tool description must steer the calling agent:
  > "Returns memory excerpts as a wall-of-text. **Always call this from inside
  > an Agent/Task subagent**, not from the main context — the wall can be 50–
  > 100kb and will bloat your context if you read it directly. Typical usage:
  > main agent dispatches `Agent(prompt='use recall_get_context to find X
  > about Y, then synthesize a concise answer')`. The subagent calls this
  > tool, reads the wall, and returns prose."
- Inputs: `search_terms`, `question`, `project` (boost), `strict_project` (filter).
- Returns: the wall-of-text + a one-line stats footer (`hit_count`, `block_count`,
  `blocks_in_wall`, `wall_chars`).

### 2. Refactor `retrieval.py`

- Rename `recall_memory()` → `build_recall_wall()` (or similar). Returns
  `(wall: str, RecallContext)` where `RecallContext` is the slimmed dataclass.
- Delete `call_subagent()`.
- Delete the `--include-partial-messages` / `--output-format stream-json`
  scaffolding it was using.
- Keep all the find/gather/render/log infrastructure — that part is fine.

### 3. Slim `RecallResult` → `RecallContext`

Drop:
- `t_subagent`, `subagent_breakdown`
- `answer` field (no longer synthesized here)

Keep:
- `hit_count`, `block_count`, `blocks_in_wall`, `wall_chars`
- `t_find_hits`, `t_gather`, `t_render`, `t_total`
- `find_hits_breakdown`

### 4. CLI: `claude-memory recall`

Decision needed:
- **Option A**: Drop `recall` from the CLI entirely. The CLI is for ingest /
  search / stats, not for end-to-end synthesis. (Cleanest.)
- **Option B**: Keep `recall` as a debugging aid, but have it call the
  Anthropic API directly with `ANTHROPIC_API_KEY`. (Useful for benchmarking
  but adds a credential dependency.)
- **Option C**: Keep `recall` calling `claude -p` for now, just for CLI
  testing — the production path through MCP no longer uses it. (Lazy, but the
  test data we already have is preserved.)

Default to A. Add B back if someone wants it for benchmarking.

### 5. Recall log

Drop subagent timing fields. The MCP server now logs only retrieval-side
phases. Subagent cost moves to the caller's Agent-tool dispatch and isn't
ours to measure (and shouldn't be — different concern).

### 6. Hook cleanup (separate, optional)

The recall side no longer cares about subagent boot, but the *main* Claude
Code session still pays for SessionStart hooks. Audit `hooks/hooks.json`:
the only hook that's load-bearing on session start is the `uv sync` venv-
heal one and the index injection. The `memory-check.sh` UserPromptSubmit
hook is also fine. Nothing to remove unless we discover something costly.

## Tool description hygiene

The new tool description has to be unambiguous. Failure mode: a calling
agent reads the description, decides "that's just a tool, I'll call it",
gets back 80kb of wall, dumps it in its context. Three lines of harm.

Mitigation:
- Lead the description with `IMPORTANT:` and the Agent-dispatch instruction
- Include a worked-example one-liner of the right pattern
- Trust the calling agent — per the user, we're not designing for dumb
  models, we're designing for the smart ones

## Validation

After implementation, run the model comparison script (`for m in opus
sonnet haiku; ...`) but instead of measuring `claude -p` subagent latency,
measure the calling agent's Agent-tool dispatch + tool call + synthesis.
Compare to the baseline numbers we already have in `recall_log.jsonl`
(versions ≤ 0.5.3).

Expected outcome:
- subagent.boot disappears from the books (was 1.5–7.5s)
- API TTFT may improve marginally if Claude Code's Agent-tool subagent
  has cache locality the cold `claude -p` doesn't
- `gen` is unchanged (model speed is model speed)
- Total recall latency drops by the boot phase + cleanup ≈ 2–8s

Net: 5–25% faster recalls, with the win concentrated on the variance tail.

## What we delete

- `call_subagent()` in `retrieval.py`
- `--output-format stream-json` / `--include-partial-messages` / `--verbose`
  scaffolding
- `subagent_breakdown` fields from `RecallResult`
- subagent log fields from `_append_recall_log`
- The `recall_memory` MCP tool registration
- Possibly the `claude-memory recall` CLI command (Option A)

## What stays

- All the find/gather/render/stitch code (works well, was never the bottleneck)
- The recall log JSONL — slimmed schema
- Project boost ranking (orthogonal feature)
- The keyword + vector hit-finding pipeline

## Open questions

- **Tool name**: `recall_get_context` is descriptive but verbose. Alternatives:
  `recall_excerpts`, `recall_raw`, `recall_window`. Pick before implementation.
- **Stats footer in tool response**: useful diagnostic for the Agent
  subagent (so it knows whether its wall was truncated), or noise that
  bloats the response? Probably useful — keep it short, one line.
- **Wall budget**: currently 120k tokens. With Agent dispatch, the subagent's
  context window is the standard 200k. May want to tighten the wall budget
  to leave more room for the question + system prompt + answer. Probably
  fine to leave at 120k for now.
- **CLI `recall` fate**: A vs B vs C above. Default A unless the user
  wants debugging-via-API.

## Rollout

- Single changeset: tool addition + retrieval refactor + log slim + CLI
  decision. Small enough to reason about together.
- Bump to 0.6.0 — this changes the MCP tool surface meaningfully.
- Update CLAUDE.md or README with one-line note about the dispatch pattern,
  in case a fresh Claude session needs to learn the convention.
- No data migration needed.
- Watch the recall log for a few days post-rollout. The latency distribution
  should compress (less variance), and the new schema will tell us if
  retrieval-side phases are now the bottleneck.
