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
                "Check conversation memory status. Returns a summary of the processed store, the "
                "time since the most recent activity in any Claude Code session file, and a "
                "per-session table of any conversations that are not yet ingested.\n\n"
                "The directive: ingest terminated sessions so their content becomes searchable. "
                "Call this at the start of every new conversation and decide whether to invoke "
                "ingest_sessions based on what you see — if the last session activity is recent, "
                "another conversation is likely still in progress and pending entries may not be "
                "terminated yet."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # search_conversation_memory is intentionally hidden from list_tools
        # as of 0.5.0 — recall_get_context covers the same ground with stitched
        # neighbor context. The handler below (_handle_search) and call_tool
        # dispatch are kept so it can be re-enabled by adding a Tool() entry
        # back here if needed. Pruning the calling-agent's choice down to one
        # memory tool also removes a routing decision it gets wrong.
        Tool(
            name="recall_get_context",
            description=(
                "IMPORTANT: ALWAYS dispatch this via an Agent/Task subagent — NEVER call from "
                "the main context. The wall can be 50–400kb of stitched conversation excerpts; "
                "reading it directly will bloat your context.\n\n"
                "WHAT THIS DOES: Searches the user's past Claude Code conversations (all "
                "projects by default). Returns a markdown wall-of-text of stitched trajectory "
                "excerpts — full per-trajectory EDU lists with neighboring trajectories on "
                "each side — matching the search terms and question.\n\n"
                "RELATIONSHIP TO THE SESSION-START INDEX: The 'Conversation memory index' "
                "block injected at session start gives one-line summaries scoped to the "
                "current project. This tool gives full stitched excerpts and searches "
                "globally. Use the index for cheap orientation; use this tool when you need "
                "the actual content, the reasoning behind a past decision, or topics not in "
                "the index.\n\n"
                "DISPATCH PATTERN:\n"
                "  Agent(\n"
                "    subagent_type=\"general-purpose\",\n"
                "    description=\"recall <topic>\",\n"
                "    prompt=\"Call recall_get_context with search_terms=[...] and \"\n"
                "           \"question='...'. Read the wall and return a concise prose \"\n"
                "           \"answer with citations to specific dates/trajectories.\"\n"
                "  )\n"
                "The subagent reads the wall and returns prose; the wall is discarded with "
                "its context. Your main context stays clean.\n\n"
                "EXAMPLES (illustrative only — DO NOT execute these literally; they are "
                "shown to demonstrate the shape of a recall call):\n"
                "  • Past technical decision:\n"
                "      search_terms=[\"pipewire\", \"filter-chain\", \"easyeffects\"]\n"
                "      question=\"why did we replace EasyEffects with a native filter-chain\"\n"
                "  • Debugging trail across sessions:\n"
                "      search_terms=[\"chromadb\", \"corruption\"]\n"
                "      question=\"how was the chromadb issue diagnosed and fixed\"\n\n"
                "ARGUMENTS:\n"
                "  - search_terms: topic keywords likely to appear in past trajectories\n"
                "  - question: the thing you actually need answered\n"
                "  - project: optional soft ranking boost (defaults to current cwd's project)\n"
                "  - strict_project: optional HARD filter — only when isolation is required\n\n"
                "Returns the wall + a one-line stats footer (hits, blocks, chars, retrieval "
                "timings)."
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
    elif name == "recall_get_context":
        return await _handle_recall(arguments)
    elif name == "ingest_sessions":
        return await _handle_ingest(arguments)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_recall(arguments: dict) -> list[TextContent]:
    from .parser import project_from_cwd
    from .retrieval import build_recall_wall

    search_terms = arguments.get("search_terms", []) or []
    question = arguments.get("question", "")
    current_project = arguments.get("project") or project_from_cwd()
    strict_project = arguments.get("strict_project")

    try:
        wall, ctx = await asyncio.to_thread(
            build_recall_wall,
            search_terms,
            question,
            current_project=current_project,
            strict_project=strict_project,
        )
    except Exception as e:
        return [TextContent(type="text", text=f"recall_get_context failed: {e}")]

    if ctx.hit_count == 0:
        scope = f"project '{strict_project}'" if strict_project else "any project"
        return [TextContent(
            type="text",
            text=f"No memories matched the search terms or question across {scope}.",
        )]

    if not wall.strip():
        return [TextContent(
            type="text",
            text=(
                f"Hits were found ({ctx.hit_count}) but their EDUs could not be retrieved "
                f"from the memory store."
            ),
        )]

    kw = ctx.find_hits_breakdown.get("keyword", 0.0)
    vec = ctx.find_hits_breakdown.get("vector", 0.0)
    footer = (
        f"\n\n---\n"
        f"(recall stats: {ctx.hit_count} hits, {ctx.block_count} blocks stitched, "
        f"{ctx.blocks_in_wall} in wall, {ctx.wall_chars} chars | "
        f"timing: total={ctx.t_total:.2f}s, "
        f"find={ctx.t_find_hits:.2f}s [kw={kw:.2f}, vec={vec:.2f}], "
        f"gather={ctx.t_gather:.2f}s, render={ctx.t_render:.3f}s)"
    )
    return [TextContent(type="text", text=wall + footer)]


def _humanize_age(seconds: float) -> str:
    """Format a duration in seconds as 'X seconds/minutes/hours/days ago'."""
    s = int(seconds)
    if s < 60:
        n = max(s, 1)
        return f"{n} second{'s' if n != 1 else ''} ago"
    if s < 3600:
        n = s // 60
        return f"{n} minute{'s' if n != 1 else ''} ago"
    if s < 86400:
        n = s // 3600
        return f"{n} hour{'s' if n != 1 else ''} ago"
    n = s // 86400
    return f"{n} day{'s' if n != 1 else ''} ago"


async def _handle_status() -> list[TextContent]:
    import time

    from .ingest import get_pending_sessions, load_ingestion_state
    from .parser import discover_sessions

    store = get_store()
    state = load_ingestion_state()
    edu_count = store.count()
    pending = get_pending_sessions()

    session_paths = discover_sessions()
    if session_paths:
        latest_mtime = max(p.stat().st_mtime for p in session_paths)
        last_session_update_str = _humanize_age(time.time() - latest_mtime)
    else:
        last_session_update_str = "no sessions found"

    lines = [
        f"Memory store: {edu_count} facts extracted from {len(state)} conversations.",
        f"Last session activity: {last_session_update_str}.",
    ]

    if pending:
        new = sum(1 for _, _, otc in pending if otc == 0)
        continued = len(pending) - new
        parts = []
        if new:
            parts.append(f"{new} new")
        if continued:
            parts.append(f"{continued} updated")
        lines.append(f"Pending: {len(pending)} conversations ({', '.join(parts)}) not yet ingested.")
        lines.append("")
        lines.append(_format_pending_table(pending, state))
    else:
        lines.append("All terminated conversations are ingested.")

    return [TextContent(type="text", text="\n".join(lines))]


def _format_pending_table(pending: list, state: dict) -> str:
    from .parser import parse_session_file

    rows = []
    for path, _h, old_turn_count in pending:
        sid = path.stem
        info = state.get(sid, {})
        session = parse_session_file(path)
        if session:
            project = session.project
            new_turns = len(session.turns) - old_turn_count
        else:
            project = info.get("project") or "(unknown)"
            new_turns = "?"
        status = "new" if old_turn_count == 0 else "updated"
        rows.append({
            "id": sid[:8],
            "project": project[:32],
            "new_turns": new_turns,
            "status": status,
        })

    rows.sort(key=lambda r: (0 if r["status"] == "new" else 1, r["id"]))

    header = f"{'session':<10} {'project':<34} {'new turns':>10} {'status':<8}"
    sep = "-" * len(header)
    body = [header, sep]
    for r in rows:
        body.append(f"{r['id']:<10} {r['project']:<34} {str(r['new_turns']):>10} {r['status']:<8}")
    return "\n".join(body)


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
