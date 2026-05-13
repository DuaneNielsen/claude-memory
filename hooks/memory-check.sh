#!/bin/bash
# UserPromptSubmit hook: once per Claude session, nudge the model to call
# ingest_sessions if there are unprocessed conversations.
#
# Per-session state via /tmp/claude-memory-prompted-<session_id>. Earlier
# versions used a single /tmp/claude-memory-prompted file that was machine-
# uptime-scoped — once set by one session it silently muted every other
# concurrent session until reboot. Keying by session_id keeps the
# "nudge once per session" semantics that the rest of the plugin assumes
# without cross-contaminating concurrent Claudes.

PAYLOAD=$(cat)
SID=$(echo "$PAYLOAD" | jq -r '.session_id // empty' 2>/dev/null)

if [ -n "$SID" ]; then
    FLAG="/tmp/claude-memory-prompted-${SID}"
    if [ -f "$FLAG" ]; then
        exit 0
    fi
    touch "$FLAG"
fi

# Don't nudge while an ingest is already in flight
if pgrep -f "claude_memory.cli ingest" > /dev/null 2>&1; then
    exit 0
fi

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
