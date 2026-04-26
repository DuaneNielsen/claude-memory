#!/usr/bin/env bash
# MCP server launcher — guarantees .venv exists before exec'ing into Python.
#
# Plugin reload spawns the MCP `command` immediately, which races with the
# SessionStart bootstrap hook on first install of a new version. Doing the
# uv sync here makes the MCP responsible for its own venv, eliminating the
# race entirely.
set -euo pipefail

cd "${CLAUDE_PLUGIN_ROOT}"

if [ ! -x .venv/bin/python ]; then
  echo "[claude-memory] bootstrapping .venv via uv sync..." >&2
  uv sync --quiet >&2
fi

exec .venv/bin/python -m claude_memory
