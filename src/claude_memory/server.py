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
        # search_conversation_memory is intentionally hidden from list_tools
        # as of 0.5.0 — recall_memory covers the same ground with stitched
        # neighbor context and subagent synthesis. The handler below
        # (_handle_search) and call_tool dispatch are kept so it can be
        # re-enabled by adding a Tool() entry back here if recall proves too
        # slow for quick lookups. Pruning the calling-agent's choice down to
        # one memory tool also removes a routing decision it gets wrong.
        Tool(
            name="recall_memory",
            description=(
                "Deep recall: given search terms and a question, an Opus subagent reads past "
                "trajectories (with neighboring context) and synthesizes an answer. Slower and "
                "more expensive than search_conversation_memory — use when you need an "
                "explanatory answer rather than a list of facts, or when the question spans "
                "multiple conversations. Search terms should be topic keywords (e.g. 'pipewire', "
                "'trajectory'); the question is the thing you actually need answered. "
                "Recall is global across projects by default; the current project is used as a "
                "soft ranking boost. Pass `strict_project` only when project isolation is "
                "genuinely needed. Returns the subagent's prose answer plus diagnostic counts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topic keywords likely to be in past trajectories (e.g. ['pipewire','audio']).",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to answer using recovered memory.",
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Optional: project name used as a soft RANKING BOOST. Results from "
                            "this project rank higher, cross-project hits still surface. "
                            "If omitted, derived from the current working directory."
                        ),
                    },
                    "strict_project": {
                        "type": "string",
                        "description": (
                            "Optional HARD FILTER: restrict recall to this project only. "
                            "Use only when isolation is explicitly required."
                        ),
                    },
                },
                "required": ["search_terms", "question"],
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
    elif name == "recall_memory":
        return await _handle_recall(arguments)
    elif name == "ingest_sessions":
        return await _handle_ingest(arguments)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_recall(arguments: dict) -> list[TextContent]:
    from .parser import project_from_cwd
    from .retrieval import recall_memory

    search_terms = arguments.get("search_terms", []) or []
    question = arguments.get("question", "")
    current_project = arguments.get("project") or project_from_cwd()
    strict_project = arguments.get("strict_project")

    try:
        result = await recall_memory(
            search_terms,
            question,
            current_project=current_project,
            strict_project=strict_project,
        )
    except Exception as e:
        return [TextContent(type="text", text=f"recall_memory failed: {e}")]

    kw = result.find_hits_breakdown.get("keyword", 0.0)
    vec = result.find_hits_breakdown.get("vector", 0.0)
    sa_spawn = result.subagent_breakdown.get("spawn", 0.0)
    sa_first = result.subagent_breakdown.get("first_event", 0.0)
    diag = (
        f"\n\n---\n"
        f"(recall diagnostics: {result.hit_count} trajectory hits, "
        f"{result.block_count} blocks stitched, {result.blocks_in_wall} included, "
        f"{result.wall_chars} chars of context)\n"
        f"(timing: total={result.t_total:.2f}s | "
        f"find={result.t_find_hits:.2f}s [kw={kw:.2f}, vec={vec:.2f}] | "
        f"gather={result.t_gather:.2f}s | render={result.t_render:.3f}s | "
        f"subagent={result.t_subagent:.2f}s [spawn={sa_spawn:.3f}, first_event={sa_first:.2f}])"
    )
    return [TextContent(type="text", text=result.answer + diag)]


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
        new = sum(1 for _, _, otc in pending if otc == 0)
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
    from .parser import project_from_cwd

    query = arguments["query"]
    current_project = arguments.get("project") or project_from_cwd()
    strict_project = arguments.get("strict_project")
    max_results = arguments.get("max_results", 10)

    store = get_store()

    if store.count() == 0:
        return [TextContent(
            type="text",
            text="Memory store is empty. Use the ingest_sessions tool to process conversations first.",
        )]

    from .query import search
    results = search(
        store,
        query,
        current_project=current_project,
        strict_project=strict_project,
        max_results=max_results,
    )

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
    from pathlib import Path

    # Check if ingestion is already running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "claude_memory.cli ingest"],
            capture_output=True,
        )
        if result.returncode == 0:
            return [TextContent(type="text", text="Ingestion is already running in the background. No action needed.")]
    except Exception:
        pass

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
    from .parser import parse_session_file
    from .extractor import count_chunks, count_chunks_incremental
    pending_refs = get_pending_sessions()
    count = len(pending_refs)

    # Estimate time: ~30s per chunk with sonnet
    total_chunks = 0
    for path, _h, old_turn_count in pending_refs:
        session = parse_session_file(path)
        if not session:
            continue
        if old_turn_count > 0:
            total_chunks += count_chunks_incremental(session, old_turn_count)
        else:
            total_chunks += count_chunks(session)
    est_minutes = max(1, (total_chunks * 30) // 60)

    lines = [f"Ingestion started in the background for {count} conversations (~{total_chunks} chunks)."]
    lines.append(f"Estimated time: ~{est_minutes} minutes.")
    lines.append("Tell the user approximately how long this will take.")
    lines.append("New memories will become searchable as they are processed.")
    lines.append("The user can continue working — this won't block anything.")

    return [TextContent(type="text", text="\n".join(lines))]


async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
