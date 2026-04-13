#!/bin/bash
# Only fire once per session — skip if already prompted
if [ -f /tmp/claude-memory-prompted ]; then
    exit 0
fi
touch /tmp/claude-memory-prompted
echo "Call the memory_status tool to check for unprocessed conversations before responding."
