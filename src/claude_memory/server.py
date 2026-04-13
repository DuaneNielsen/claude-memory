"""MCP server exposing conversation memory search to Claude Code."""

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

log = logging.getLogger(__name__)

server = Server("claude-memory")
_store = None


def get_store():
    global _store
    if _store is None:
        from .store import MemoryStore
        _store = MemoryStore()
    return _store


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="memory_status",
            description=(
                "Check conversation memory status — how many sessions are ingested vs pending. "
                "IMPORTANT: Call this at the START of every new conversation. "
                "This system extracts discrete facts from past conversations so they can be searched later. "
                "If a small number of sessions are pending (≤5), silently ingest them in the background "
                "using the ingest_sessions tool without telling the user. "
                "If many sessions are pending (>5), tell the user and ask before ingesting — "
                "explain that you have unprocessed conversations and would like to extract memories from them "
                "so you can recall past work. Keep it brief and natural. "
                "If nothing is pending, proceed normally."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="search_conversation_memory",
            description=(
                "Search past Claude Code conversations for relevant context. "
                "Search when the user references past work, asks about previous decisions, "
                "or when the task involves systems that may have been configured or debugged before. "
                "Also search proactively when context seems to be missing. "
                "The query is fast and a miss costs nothing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter to a specific project name (e.g. 'primesignal', 'frigate')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest_sessions",
            description=(
                "Process new or changed Claude Code conversations into searchable memory. "
                "Extracts discrete facts from conversations and stores them for later recall. "
                "For ≤5 pending conversations, call this silently in the background. "
                "For more, ask the user first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Claude model to use for extraction (default: sonnet)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Re-ingest all sessions from scratch (default: false)",
                        "default": False,
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "memory_status":
        return await _handle_status()
    elif name == "search_conversation_memory":
        return await _handle_search(arguments)
    elif name == "ingest_sessions":
        return await _handle_ingest(arguments)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_status() -> list[TextContent]:
    from .ingest import get_pending_sessions, load_ingestion_state

    store = get_store()
    state = load_ingestion_state()
    edu_count = store.count()
    pending = get_pending_sessions()

    lines = [
        f"Memory store: {edu_count} facts extracted from {len(state)} conversations.",
    ]

    if pending:
        new = sum(1 for _, otc in pending if otc == 0)
        continued = len(pending) - new
        parts = []
        if new:
            parts.append(f"{new} new")
        if continued:
            parts.append(f"{continued} updated")
        lines.append(f"Pending: {len(pending)} conversations ({', '.join(parts)}) have not been processed yet.")
        lines.append("Processing extracts searchable facts from past conversations so you can recall prior work.")
    else:
        lines.append("All conversations are up to date.")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_search(arguments: dict) -> list[TextContent]:
    query = arguments["query"]
    project = arguments.get("project")
    max_results = arguments.get("max_results", 10)

    store = get_store()

    if store.count() == 0:
        return [TextContent(
            type="text",
            text="Memory store is empty. Use the ingest_sessions tool to process conversations first.",
        )]

    from .query import search
    results = search(store, query, project=project, max_results=max_results)

    if not results:
        return [TextContent(type="text", text="No relevant memories found.")]

    lines = [f"Found {len(results)} relevant memories:\n"]
    for i, r in enumerate(results, 1):
        date = r.timestamp.strftime("%Y-%m-%d")
        lines.append(f"{i}. [{r.project}, {date}] {r.text}")
        lines.append(f"   (similarity={r.similarity:.3f}, recency={r.recency_weight:.3f}, score={r.score:.3f})")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_ingest(arguments: dict) -> list[TextContent]:
    import subprocess
    import sys
    from pathlib import Path

    model = arguments.get("model")
    force = arguments.get("force", False)

    # Find the python executable in our venv
    venv_python = str(Path(__file__).parent.parent.parent / ".venv" / "bin" / "python")

    cmd = [venv_python, "-m", "claude_memory.cli", "ingest"]
    if model:
        cmd.extend(["--model", model])
    if force:
        cmd.append("--force")

    try:
        # Launch as detached background process
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return [TextContent(type="text", text=f"Failed to start ingestion: {e}")]

    from .ingest import get_pending_sessions
    pending = get_pending_sessions()
    count = len(pending)

    lines = [f"Ingestion started in the background for {count} conversations."]
    lines.append("New memories will become searchable as they are processed.")
    lines.append("The user can continue working — this won't block anything.")

    return [TextContent(type="text", text="\n".join(lines))]


async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
