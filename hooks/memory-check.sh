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

if [ "$STATUS" -gt 0 ] 2>/dev/null; then
    # Return JSON with both a system message (user sees) and context (Claude sees)
    cat <<ENDJSON
{
  "systemMessage": "[claude-memory] $STATUS unprocessed conversations detected.",
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "IMPORTANT: There are $STATUS unprocessed conversations. If this is 5 or fewer, silently call the ingest_sessions tool in the background. If more than 5, briefly tell the user you have $STATUS unprocessed conversations and ask if they'd like you to process them so you can recall past work."
  }
}
ENDJSON
fi
