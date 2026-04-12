"""MCP server exposing conversation memory search to Claude Code."""

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .query import search
from .store import MemoryStore

log = logging.getLogger(__name__)

server = Server("claude-memory")
_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_conversation_memory",
            description=(
                "Search past Claude Code conversations for relevant context. "
                "Use this when the user references past work, asks 'did we discuss...', "
                "or when historical context would help with the current task."
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "search_conversation_memory":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    query = arguments["query"]
    project = arguments.get("project")
    max_results = arguments.get("max_results", 10)

    store = get_store()

    if store.count() == 0:
        return [TextContent(
            type="text",
            text="Memory store is empty. Run `claude-memory ingest` first.",
        )]

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


async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
