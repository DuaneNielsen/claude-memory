#!/bin/bash
# Triggered on SessionStart with source=clear.
# Ingests all pending sessions for this user, EXCLUDING the new (post-clear)
# session whose ID is in the SessionStart payload — that one's still active.

PAYLOAD=$(cat)
NEW_SESSION_ID=$(echo "$PAYLOAD" | jq -r '.session_id // empty' 2>/dev/null)

if [ -z "$NEW_SESSION_ID" ]; then
    exit 0
fi

# Avoid stomping on a concurrent ingest run (e.g. one already kicked off
# by memory-check.sh on UserPromptSubmit).
if pgrep -f "claude_memory.cli ingest" > /dev/null 2>&1; then
    exit 0
fi

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="${CLAUDE_PLUGIN_DATA:-/tmp}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/clear-ingest.log"

# Belt-and-suspenders: ensure venv exists. The unmatched SessionStart entry
# also runs uv sync if needed, but ordering between entries isn't guaranteed.
if [ ! -x "$PLUGIN_ROOT/.venv/bin/python" ]; then
    (cd "$PLUGIN_ROOT" && uv sync --quiet) >> "$LOG" 2>&1
fi

{
    echo "=== $(date -Iseconds) clear-ingest fired (excluding $NEW_SESSION_ID) ==="
} >> "$LOG"

# Fire-and-forget. nohup + & + disown detaches fully so SessionStart returns.
nohup "$PLUGIN_ROOT/.venv/bin/python" -m claude_memory.cli ingest \
    --exclude "$NEW_SESSION_ID" >> "$LOG" 2>&1 &
disown 2>/dev/null || true

exit 0
