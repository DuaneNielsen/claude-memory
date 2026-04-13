#!/bin/bash
# Only fire once per session
if [ -f /tmp/claude-memory-prompted ]; then
    exit 0
fi
touch /tmp/claude-memory-prompted

# Run the actual status check
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
STATUS=$("$PLUGIN_ROOT/.venv/bin/python" -c "
from claude_memory.ingest import get_pending_sessions
pending = get_pending_sessions()
if pending:
    print(len(pending))
else:
    print(0)
" 2>/dev/null)

if [ "$STATUS" -gt 5 ] 2>/dev/null; then
    # Many pending — show to user and ask before ingesting
    cat <<ENDJSON
{
  "systemMessage": "[claude-memory] $STATUS unprocessed conversations detected.",
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "There are $STATUS unprocessed conversations. Briefly tell the user and ask if they'd like you to process them so you can recall past work."
  }
}
ENDJSON
elif [ "$STATUS" -gt 0 ] 2>/dev/null; then
    # Few pending — silently ingest, no user-visible message
    cat <<ENDJSON
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "There are $STATUS unprocessed conversations. Silently call the ingest_sessions tool in the background without telling the user."
  }
}
ENDJSON
fi
