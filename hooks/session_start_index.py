#!/usr/bin/env python3
"""SessionStart hook: inject the project memory index into Claude's context.

Reads ~/.local/share/claude-memory/indices/<project>.md if it exists and
emits it as hookSpecificOutput.additionalContext. No LLM calls, no DB
queries — just a file read, so latency stays well under 50ms.

Silent exit when no index file exists (e.g. a project that hasn't been
ingested yet). Failures are swallowed and logged to stderr so a broken
hook never blocks a session.
"""

import json
import os
import sys
from pathlib import Path


def _project_from_cwd(cwd: Path) -> str:
    """Mirror claude_memory.parser.project_from_cwd without importing it.

    Keeping this hook free of package imports means it runs fast and doesn't
    depend on the plugin's venv being initialized.
    """
    encoded = "-" + str(cwd.resolve()).lstrip("/").replace("/", "-")
    parts = encoded.strip("-").split("-")
    if len(parts) >= 2 and parts[0] == "home":
        remainder = "-".join(parts[2:]) if len(parts) > 2 else "home"
        return remainder or "home"
    return encoded


def main() -> int:
    try:
        cwd = Path(os.environ.get("PWD") or os.getcwd())
        project = _project_from_cwd(cwd)

        indices_dir = Path.home() / ".local" / "share" / "claude-memory" / "indices"
        index_path = indices_dir / f"{project}.md"

        if not index_path.exists():
            return 0

        content = index_path.read_text()
        if not content.strip():
            return 0

        # Prepend a stable, distinctively titled heading so Claude recognizes
        # this block as a discrete corpus and not loose context. The wrapper
        # text is byte-identical across runs so it doesn't break prompt caching.
        #
        # Catalog framing: most sections below are POINTERS, not content. The
        # full text of past memories lives in the recall store; this index
        # advertises what's available so Claude knows what to fetch.
        wrapped = (
            f"# Conversation memory index — project `{project}`\n\n"
            "The sections below catalog what memories are available for this "
            "project. Most are pointers — call the `recall_get_context` MCP "
            "tool (via an Agent/Task subagent) to fetch the actual content "
            "for any topic of interest.\n\n"
            "The **Preferences** section is the exception: those shape "
            "behavior and apply immediately, so the full text is included.\n\n"
            "---\n\n"
            + content
        )

        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": wrapped,
            }
        }
        json.dump(output, sys.stdout)
    except Exception as e:
        print(f"[claude-memory session_start_index] error: {e}", file=sys.stderr)
        return 0  # never fail the hook
    return 0


if __name__ == "__main__":
    sys.exit(main())
