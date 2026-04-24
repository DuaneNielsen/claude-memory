"""CLI entry point for claude-memory."""

import argparse
import asyncio
import logging
import sys

from .config import DATA_DIR, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(prog="claude-memory", description="Claude Code conversation memory")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest Claude Code sessions")
    ingest_p.add_argument("--force", action="store_true", help="Re-ingest all sessions")
    ingest_p.add_argument("--model", default=None, help=f"Claude model (default: {DEFAULT_MODEL})")
    ingest_p.add_argument("--concurrency", type=int, default=2, help="Concurrent extraction requests (default: 2)")

    # search
    search_p = sub.add_parser("search", help="Search conversation memory")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument(
        "--project",
        help="Project name used as a soft ranking boost (default: derived from cwd). "
             "Cross-project hits still surface.",
    )
    search_p.add_argument(
        "--strict-project",
        help="Hard-filter results to this project only (overrides boost behavior).",
    )
    search_p.add_argument("--max-results", type=int, default=10)

    # stats
    sub.add_parser("stats", help="Show memory store statistics")

    # sessions
    sessions_p = sub.add_parser("sessions", help="List ingested sessions with details")
    sessions_p.add_argument("--project", help="Filter to project")
    sessions_p.add_argument("--sort", choices=["date", "edus", "words"], default="date")

    # dump
    dump_p = sub.add_parser("dump", help="Dump all EDUs as readable text or JSON")
    dump_p.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    dump_p.add_argument("--project", help="Filter to project")

    # reset
    reset_p = sub.add_parser("reset", help="Clear ingestion state and/or stored EDUs")
    reset_p.add_argument("--session", help="Reset a specific session ID (prefix match)")
    reset_p.add_argument("--project", help="Reset all sessions for a project")
    reset_p.add_argument("--all", action="store_true", help="Reset everything")
    reset_p.add_argument("--partial", action="store_true", help="Reset all partially-ingested sessions (for full re-extraction)")
    reset_p.add_argument("--state-only", action="store_true", help="Only clear ingestion state, keep EDUs in ChromaDB")

    # reindex
    reindex_p = sub.add_parser("reindex", help="Rebuild project memory indices")
    reindex_p.add_argument("--project", help="Rebuild one project's index (default: all)")

    # recall (subagent deep-recall, for testing)
    recall_p = sub.add_parser("recall", help="Deep-recall via Opus subagent")
    recall_p.add_argument("question", help="The question to answer from memory")
    recall_p.add_argument("--terms", nargs="*", default=[], help="Keyword search terms")
    recall_p.add_argument(
        "--project",
        help="Project name used as a soft ranking boost (default: derived from cwd). "
             "Cross-project hits still surface.",
    )
    recall_p.add_argument(
        "--strict-project",
        help="Hard-filter recall to this project only (overrides boost behavior).",
    )
    recall_p.add_argument("--model", default="opus", help="Subagent model (default: opus)")

    # serve (MCP server)
    sub.add_parser("serve", help="Run MCP server (stdio)")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    # Prefix non-INFO messages with the level name so warnings stand out amid
    # the progress bar output. INFO stays unprefixed for readability.
    class _LevelFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno >= logging.WARNING:
                return f"[{record.levelname}] {record.getMessage()}"
            return record.getMessage()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LevelFormatter())
    logging.basicConfig(level=level, handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    if args.command == "ingest":
        from .ingest import ingest_all, SchemaVersionMismatch
        try:
            stats = asyncio.run(ingest_all(model=args.model, force=args.force, concurrency=args.concurrency))
        except SchemaVersionMismatch as e:
            print(f"Error: {e}")
            sys.exit(1)
        prefix = "Rate-limited — stopped for later resume" if stats.get("rate_limited") else "Ingested"
        print(
            f"{prefix}: {stats.get('trajectories', 0)} trajectories / "
            f"{stats['edus']} EDUs from {stats['sessions']} sessions."
        )
        if stats.get("pending"):
            print(f"{stats['pending']} session(s) left pending for retry — re-run `claude-memory ingest` to resume.")

    elif args.command == "search":
        from .parser import project_from_cwd
        from .query import search
        from .store import MemoryStore
        store = MemoryStore()
        current_project = args.project or project_from_cwd()
        results = search(
            store,
            args.query,
            current_project=current_project,
            strict_project=args.strict_project,
            max_results=args.max_results,
        )
        if not results:
            print("No results found.")
        else:
            for i, r in enumerate(results, 1):
                date = r.timestamp.strftime("%Y-%m-%d")
                boost_marker = " *boost*" if r.project_boost > 1.0 else ""
                print(f"{i}. [{r.project}{boost_marker}, {date}] {r.text}")
                print(
                    f"   score={r.score:.3f} (sim={r.similarity:.3f}, "
                    f"recency={r.recency_weight:.3f}, boost={r.project_boost:.2f})"
                )
                print()

    elif args.command == "stats":
        from .store import MemoryStore
        from .ingest import load_ingestion_state
        store = MemoryStore()
        state = load_ingestion_state()
        count = store.count()
        print(f"EDUs in store: {count}")
        print(f"Sessions ingested: {len(state)}")
        if state:
            projects = {}
            for info in state.values():
                p = info.get("project", "unknown")
                projects[p] = projects.get(p, 0) + 1
            print("By project:")
            for p, c in sorted(projects.items(), key=lambda x: -x[1]):
                print(f"  {p}: {c} sessions")

    elif args.command == "sessions":
        from .ingest import load_ingestion_state
        from .parser import parse_session_file
        from pathlib import Path
        state = load_ingestion_state()
        rows = []
        for sid, info in state.items():
            project = info.get("project", "?")
            if args.project and project != args.project:
                continue
            edu_count = info.get("edu_count", 0)
            date = (info.get("timestamp") or "")[:10]
            # Count words from the actual session file
            path = Path(info.get("file_path", ""))
            words = 0
            turns = 0
            if path.exists():
                session = parse_session_file(path)
                if session:
                    turns = len(session.turns)
                    words = sum(len(t.text.split()) for t in session.turns)
            partial = "yes" if info.get("partial") else ""
            rows.append((date, project, sid[:8], turns, words, edu_count, partial))

        sort_key = {"date": 0, "edus": 5, "words": 4}[args.sort]
        rows.sort(key=lambda r: r[sort_key], reverse=(args.sort != "date"))

        from rich.console import Console
        from rich.table import Table
        table = Table(title="Ingested Sessions")
        table.add_column("Date", style="cyan")
        table.add_column("Project", style="green")
        table.add_column("Session")
        table.add_column("Turns", justify="right")
        table.add_column("Words", justify="right")
        table.add_column("EDUs", justify="right", style="yellow")
        table.add_column("Partial", style="red")
        for date, project, sid, turns, words, edus, partial in rows:
            table.add_row(date, project, sid, str(turns), f"{words:,}", str(edus), partial)
        table.add_section()
        table.add_row(
            "Total", "", "",
            str(sum(r[3] for r in rows)),
            f"{sum(r[4] for r in rows):,}",
            str(sum(r[5] for r in rows)),
            "",
            style="bold",
        )
        Console().print(table)

    elif args.command == "dump":
        import json as json_mod
        from .store import MemoryStore
        store = MemoryStore()
        where = {"project": args.project} if args.project else None
        data = store.collection.get(include=["documents", "metadatas"], where=where)
        if not data["ids"]:
            print("No EDUs in store.")
        elif args.as_json:
            entries = []
            for doc, meta in zip(data["documents"], data["metadatas"]):
                entries.append({"text": doc, **meta})
            entries.sort(key=lambda e: e.get("timestamp", ""))
            print(json_mod.dumps(entries, indent=2))
        else:
            pairs = list(zip(data["documents"], data["metadatas"]))
            pairs.sort(key=lambda p: p[1].get("timestamp", ""))
            for doc, meta in pairs:
                date = meta.get("timestamp", "")[:10]
                project = meta.get("project", "?")
                print(f"[{project}, {date}] {doc}")
                print()

    elif args.command == "reset":
        from .config import INGESTION_STATE_FILE, TRAJECTORIES_DB
        from .ingest import load_ingestion_state, save_ingestion_state, SchemaVersionMismatch
        from .store import MemoryStore

        if not args.all and not args.session and not args.project and not args.partial:
            print("Specify --all, --session <id>, --project <name>, or --partial")
            sys.exit(1)

        # --all is the escape hatch: it MUST work even if state/DB are in an
        # incompatible format from an older version. Nuke everything wholesale
        # without trying to read the state file.
        if args.all:
            # Delete state file so the next load returns empty
            if INGESTION_STATE_FILE.exists():
                INGESTION_STATE_FILE.unlink()
            # Delete trajectory DB entirely (and its WAL/SHM companions)
            for p in [TRAJECTORIES_DB, TRAJECTORIES_DB.with_suffix(".db-wal"),
                      TRAJECTORIES_DB.with_suffix(".db-shm")]:
                if p.exists():
                    p.unlink()
            # Clear ChromaDB collection (recreates fresh)
            store = MemoryStore()
            data = store.collection.get()
            if data["ids"]:
                store.collection.delete(ids=data["ids"])
            print("Reset all state, trajectories, and EDUs. Run `claude-memory ingest` to reprocess.")
            sys.exit(0)

        # Scoped reset — needs state to identify targets, so we load it.
        # If version mismatch, instruct the user to use --all.
        try:
            state = load_ingestion_state()
        except SchemaVersionMismatch as e:
            print(f"Error: {e}")
            print("Use `claude-memory reset --all` to wipe and start over.")
            sys.exit(1)

        from .trajectories import TrajectoryStore
        store = MemoryStore()
        traj_store = TrajectoryStore()

        # Find matching session IDs
        if args.partial:
            targets = [sid for sid, info in state.items() if info.get("partial")]
        elif args.project:
            targets = [sid for sid, info in state.items() if info.get("project") == args.project]
        else:
            targets = [sid for sid in state if sid.startswith(args.session)]

        if not targets:
            print("No matching sessions found.")
            sys.exit(0)

        # Clear EDUs + trajectories unless --state-only
        if not args.state_only:
            for sid in targets:
                deleted_edus = store.delete_session(sid)
                deleted_trajs = traj_store.delete_session(sid)
                if deleted_edus or deleted_trajs:
                    print(f"Deleted {deleted_trajs} trajectories / {deleted_edus} EDUs for {sid[:8]}")

        # Clear ingestion state
        for sid in targets:
            del state[sid]
        save_ingestion_state(state)

        print(f"Reset {len(targets)} session(s). Run `claude-memory ingest` to reprocess.")

    elif args.command == "recall":
        from .parser import project_from_cwd
        from .retrieval import recall_memory
        current_project = args.project or project_from_cwd()
        result = asyncio.run(recall_memory(
            search_terms=args.terms,
            question=args.question,
            current_project=current_project,
            strict_project=args.strict_project,
            model=args.model,
        ))
        print(result.answer)
        kw = result.find_hits_breakdown.get("keyword", 0.0)
        vec = result.find_hits_breakdown.get("vector", 0.0)
        print(
            f"\n---\n(diagnostics: {result.hit_count} hits, "
            f"{result.block_count} blocks, {result.blocks_in_wall} in wall, "
            f"{result.wall_chars} chars)"
        )
        print(
            f"(timing: total={result.t_total:.2f}s | "
            f"find={result.t_find_hits:.2f}s [kw={kw:.2f}, vec={vec:.2f}] | "
            f"gather={result.t_gather:.2f}s | render={result.t_render:.3f}s | "
            f"subagent={result.t_subagent:.2f}s)"
        )

    elif args.command == "reindex":
        from .index_builder import write_index
        from .trajectories import TrajectoryStore
        traj_store = TrajectoryStore()
        projects = [args.project] if args.project else traj_store.list_projects()
        if not projects:
            print("No projects found in trajectory store.")
            sys.exit(0)
        for project in projects:
            try:
                path = write_index(project, traj_store=traj_store)
                print(f"Wrote {path}")
            except Exception as e:
                print(f"Failed to build index for {project}: {e}")

    elif args.command == "serve":
        from .server import main as serve_main
        serve_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
