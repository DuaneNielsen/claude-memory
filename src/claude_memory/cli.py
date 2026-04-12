"""CLI entry point for claude-memory."""

import argparse
import asyncio
import logging
import sys

from .config import DATA_DIR, LLM_BASE_URL


def main():
    parser = argparse.ArgumentParser(prog="claude-memory", description="Claude Code conversation memory")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest Claude Code sessions")
    ingest_p.add_argument("--force", action="store_true", help="Re-ingest all sessions")
    ingest_p.add_argument("--llm-url", default=None, help=f"LLM endpoint (default: {LLM_BASE_URL})")
    ingest_p.add_argument("--concurrency", type=int, default=2, help="Concurrent LLM requests (default: 2)")

    # search
    search_p = sub.add_parser("search", help="Search conversation memory")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--project", help="Filter to project")
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
    reset_p.add_argument("--state-only", action="store_true", help="Only clear ingestion state, keep EDUs in ChromaDB")

    # serve (MCP server)
    sub.add_parser("serve", help="Run MCP server (stdio)")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    if args.command == "ingest":
        from .ingest import ingest_all
        stats = asyncio.run(ingest_all(base_url=args.llm_url, force=args.force, concurrency=args.concurrency))
        print(f"Ingested {stats['edus']} EDUs from {stats['sessions']} sessions.")

    elif args.command == "search":
        from .query import search
        from .store import MemoryStore
        store = MemoryStore()
        results = search(store, args.query, project=args.project, max_results=args.max_results)
        if not results:
            print("No results found.")
        else:
            for i, r in enumerate(results, 1):
                date = r.timestamp.strftime("%Y-%m-%d")
                print(f"{i}. [{r.project}, {date}] {r.text}")
                print(f"   score={r.score:.3f} (sim={r.similarity:.3f}, recency={r.recency_weight:.3f})")
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
            rows.append((date, project, sid[:8], turns, words, edu_count))

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
        for date, project, sid, turns, words, edus in rows:
            table.add_row(date, project, sid, str(turns), f"{words:,}", str(edus))
        table.add_section()
        table.add_row(
            "Total", "", "",
            str(sum(r[3] for r in rows)),
            f"{sum(r[4] for r in rows):,}",
            str(sum(r[5] for r in rows)),
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
        from .ingest import load_ingestion_state, save_ingestion_state
        from .store import MemoryStore
        state = load_ingestion_state()
        store = MemoryStore()

        if not args.all and not args.session and not args.project:
            print("Specify --all, --session <id>, or --project <name>")
            sys.exit(1)

        # Find matching session IDs
        if args.all:
            targets = list(state.keys())
        elif args.project:
            targets = [sid for sid, info in state.items() if info.get("project") == args.project]
        else:
            targets = [sid for sid in state if sid.startswith(args.session)]

        if not targets:
            print("No matching sessions found.")
            sys.exit(0)

        # Clear EDUs from ChromaDB unless --state-only
        if not args.state_only:
            for sid in targets:
                deleted = store.delete_session(sid)
                if deleted:
                    print(f"Deleted {deleted} EDUs for {sid[:8]}")

        # Clear ingestion state
        for sid in targets:
            del state[sid]
        save_ingestion_state(state)

        print(f"Reset {len(targets)} session(s). Run `claude-memory ingest` to reprocess.")

    elif args.command == "serve":
        from .server import main as serve_main
        serve_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
