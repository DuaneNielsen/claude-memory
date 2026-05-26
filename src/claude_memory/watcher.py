"""Background watcher: inotify on ~/.claude/projects/ → debounced ingest_all().

Replaces the SessionStart `/clear` hook + UserPromptSubmit nudge + MCP
ingest_sessions tool. Runs as a long-lived user-level systemd service.

Loop:
  1. Watch the projects root for new project subdirs (IN_CREATE / IN_MOVED_TO).
  2. Watch each project subdir for *.jsonl writes (IN_MODIFY / IN_CREATE / IN_MOVED_TO).
  3. Any JSONL event bumps `last_change`.
  4. Once `now - last_change > QUIET_SECONDS` AND there is pending work, run a
     sweep via ingest_all(). On startup, run one sweep immediately if backlog
     exists (drains the post-install backlog without waiting for a fresh event).
  5. SIGTERM exits cleanly after the current sweep finishes.
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

from inotify_simple import INotify, flags

from .config import CLAUDE_PROJECTS_DIR
from .ingest import SchemaVersionMismatch, get_pending_sessions, ingest_all

log = logging.getLogger(__name__)

WATCH_FLAGS = flags.MODIFY | flags.CREATE | flags.MOVED_TO
DEFAULT_QUIET_SECONDS = 45
POLL_TIMEOUT_MS = 1000


class Watcher:
    def __init__(self, quiet_seconds: int = DEFAULT_QUIET_SECONDS):
        self.quiet_seconds = quiet_seconds
        self.inotify = INotify()
        self.root_wd: int | None = None
        self.project_wds: dict[int, Path] = {}
        self.last_change = time.monotonic()
        self.pending_changes = False
        self._stop = False

    def _add_project_watch(self, path: Path) -> None:
        if path in self.project_wds.values():
            return
        try:
            wd = self.inotify.add_watch(str(path), WATCH_FLAGS)
        except OSError as e:
            log.warning("Could not watch %s: %s", path, e)
            return
        self.project_wds[wd] = path
        log.debug("Watching project dir: %s", path)

    def _setup_watches(self) -> None:
        CLAUDE_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        self.root_wd = self.inotify.add_watch(str(CLAUDE_PROJECTS_DIR), WATCH_FLAGS)
        for entry in CLAUDE_PROJECTS_DIR.iterdir():
            if entry.is_dir():
                self._add_project_watch(entry)
        log.info(
            "Watching %s + %d project dirs", CLAUDE_PROJECTS_DIR, len(self.project_wds)
        )

    def _process_events(self) -> None:
        try:
            events = self.inotify.read(timeout=POLL_TIMEOUT_MS)
        except InterruptedError:
            return
        for event in events:
            if event.wd == self.root_wd:
                new_path = CLAUDE_PROJECTS_DIR / event.name
                if new_path.is_dir():
                    self._add_project_watch(new_path)
            else:
                if event.name.endswith(".jsonl"):
                    self.last_change = time.monotonic()
                    self.pending_changes = True

    def _ready_to_sweep(self) -> bool:
        if not self.pending_changes:
            return False
        return (time.monotonic() - self.last_change) > self.quiet_seconds

    def _run_sweep(self) -> None:
        try:
            pending = get_pending_sessions()
        except SchemaVersionMismatch as e:
            log.error("Schema mismatch — refusing to ingest: %s", e)
            self.pending_changes = False
            return
        if not pending:
            log.debug("Sweep skipped: no pending sessions")
            self.pending_changes = False
            return
        log.info("Sweep starting: %d pending session(s)", len(pending))
        t0 = time.time()
        try:
            stats = asyncio.run(ingest_all())
        except Exception:
            log.exception("Sweep failed")
            return
        elapsed = time.time() - t0
        log.info(
            "Sweep done in %.0fs: %d trajectories / %d EDUs across %d sessions"
            "%s",
            elapsed,
            stats.get("trajectories", 0),
            stats.get("edus", 0),
            stats.get("sessions", 0),
            " (rate-limited)" if stats.get("rate_limited") else "",
        )
        self.pending_changes = False

    def _initial_backlog_sweep(self) -> None:
        try:
            pending = get_pending_sessions()
        except SchemaVersionMismatch as e:
            log.error("Schema mismatch on startup — refusing to ingest: %s", e)
            return
        if pending:
            log.info("Backlog on startup: %d session(s) — running initial sweep",
                     len(pending))
            self.pending_changes = True
            self.last_change = time.monotonic() - self.quiet_seconds - 1
            self._run_sweep()
        else:
            log.info("No backlog on startup")

    def stop(self) -> None:
        log.info("Stop requested — will exit after current iteration")
        self._stop = True

    def run(self) -> None:
        self._setup_watches()
        self._initial_backlog_sweep()
        while not self._stop:
            self._process_events()
            if self._ready_to_sweep():
                self._run_sweep()
        log.info("Watcher exited cleanly")


def run(quiet_seconds: int = DEFAULT_QUIET_SECONDS, once: bool = False) -> int:
    """Entry point used by the `claude-memory watcher` CLI subcommand."""
    if once:
        try:
            stats = asyncio.run(ingest_all())
        except SchemaVersionMismatch as e:
            log.error("%s", e)
            return 1
        log.info(
            "One-shot sweep: %d trajectories / %d EDUs from %d sessions",
            stats.get("trajectories", 0),
            stats.get("edus", 0),
            stats.get("sessions", 0),
        )
        return 0

    watcher = Watcher(quiet_seconds=quiet_seconds)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: watcher.stop())
    try:
        watcher.run()
    except KeyboardInterrupt:
        watcher.stop()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run())
