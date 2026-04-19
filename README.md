# claude-memory

Local conversation memory for Claude Code. Parses JSONL session files, extracts atomic facts (EDUs) via the Claude API, embeds them in ChromaDB, and exposes semantic search through an MCP server.

## How it works

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
      "mcp__plugin_claude-memory_claude-memory__search_conversation_memory",
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

When installed as a plugin, the MCP server starts automatically. It exposes a `search_conversation_memory` tool that Claude Code can call to look up prior work.

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
