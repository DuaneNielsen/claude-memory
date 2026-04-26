# claude-memory

> 📖 [The Building Blocks of an Agent Memory System](https://proximal.hashnode.dev/the-building-blocks-of-an-agent-memory-system) — how this system is designed and how it works under the hood.

Local conversation memory for Claude Code. Parses JSONL session files, extracts atomic facts (EDUs) via the Claude API, embeds them in ChromaDB, and exposes semantic search through an MCP server.

## How it works

```mermaid
flowchart TB
    S["Claude Code session<br/>~/.claude/projects/*.jsonl"]

    subgraph WRITE["Ingest pipeline"]
      direction TB
      P["Parse turns"]
      E["Extract EDUs<br/>(atomic facts via Claude API)"]
      EM["Embed<br/>nomic-embed-text-v1.5 · 768d"]
      IX["Per-project index<br/>(one-line summaries)"]
      P --> E --> EM
      E --> IX
    end

    DB[("ChromaDB<br/>cosine + recency decay")]
    INDEX[("indices/&lt;project&gt;.md")]

    subgraph READ["Read path (in a new session)"]
      direction TB
      H1["SessionStart hook<br/>injects index into context"]
      MS["MCP: memory_status<br/>(pending/ingesting summary)"]
      RC["MCP: recall_get_context<br/>(stitched excerpts, via subagent)"]
    end

    C(["Active Claude conversation"])

    S --> P
    EM --> DB
    IX --> INDEX
    INDEX --> H1
    S --> MS
    DB --> RC
    H1 --> C
    MS --> C
    RC --> C
```

1. **Parse** — reads Claude Code session files from `~/.claude/projects/`
2. **Extract** — sends conversation turns to the Claude API (via `claude` CLI) which decomposes them into Elementary Discourse Units (EDUs) — self-contained atomic facts
3. **Embed** — encodes EDUs with `nomic-ai/nomic-embed-text-v1.5` (768d) and stores them in ChromaDB with cosine similarity
4. **Search** — recency-weighted retrieval: `score = similarity * e^(-0.007 * days_ago)`

Resumed sessions are handled incrementally — existing EDUs are loaded as context, and only new turns are processed.

## Install

```bash
claude plugin marketplace add DuaneNielsen/claude-memory
claude plugin install claude-memory@duane-claude-plugins
```

The plugin auto-creates a `.venv` and installs dependencies on first session start via `uv sync`. The MCP server and hooks are registered automatically.

To allow the plugin's MCP tools to run without permission prompts, add these to your `~/.claude/settings.json` allow list:

```json
{
  "permissions": {
    "allow": [
      "mcp__plugin_claude-memory_claude-memory__ingest_sessions",
      "mcp__plugin_claude-memory_claude-memory__recall_get_context",
      "mcp__plugin_claude-memory_claude-memory__memory_status"
    ]
  }
}
```

Requires `claude` CLI to be installed and authenticated (Claude Code). EDU extraction uses the Claude API via the CLI — no local LLM needed.

## CLI

```bash
# Ingest all new/changed sessions
claude-memory ingest

# Re-ingest everything from scratch
claude-memory ingest --force

# Use a specific model (default: sonnet)
claude-memory ingest --model opus

# Tune concurrency
claude-memory ingest --concurrency 4

# Search memories
claude-memory search "pipewire filter chain config"
claude-memory search "auth middleware" --project primesignal

# Show stats
claude-memory stats

# List sessions with turn/word/EDU counts
claude-memory sessions
claude-memory sessions --sort edus
claude-memory sessions --project home

# Dump all EDUs
claude-memory dump
claude-memory dump --json --project frigate

# Reset ingestion state
claude-memory reset --session abc123    # prefix match
claude-memory reset --project home
claude-memory reset --partial           # reset all incrementally-ingested sessions
claude-memory reset --all
claude-memory reset --all --state-only  # keep EDUs, clear state (forces re-check)

# Run MCP server (stdio)
claude-memory serve
```

## MCP server

When installed as a plugin, the MCP server starts automatically and exposes:

- `memory_status` — sanity check at session start; surfaces pending sessions.
- `ingest_sessions` — process new/changed conversations into searchable memory.
- `recall_get_context` — given search terms + a question, returns a stitched
  wall-of-text of relevant past trajectories with neighboring context.

**Dispatch convention for `recall_get_context`:** the wall can be tens of
kilobytes. The main agent should dispatch the call into an `Agent`/`Task`
subagent so the wall lives in the subagent's context, not yours, and the
subagent returns synthesized prose.

## References

The EDU extraction approach is based on **EMem** ("A Simple Yet Strong Baseline for Long-Term Conversational Memory", [arXiv:2511.17208](https://arxiv.org/abs/2511.17208)), which decomposes conversations into atomic Elementary Discourse Units and retrieves them via dense similarity with recency weighting — achieving 0.780 on LoCoMo using only 738 tokens of context vs 23,653 for full-context baselines.

## Data storage

- ChromaDB: `~/.local/share/claude-memory/chromadb/`
- Ingestion state: `~/.local/share/claude-memory/ingested_sessions.json`

## Session retention

Claude Code deletes sessions older than 30 days by default. To keep history longer, add to `~/.claude/settings.json`:

```json
{
  "cleanupPeriodDays": 3650
}
```
