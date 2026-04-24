"""SQLite-backed storage for topic trajectories and their keywords.

A trajectory is a contiguous range of turns within a session covering a
single topic. EDUs extracted from those turns carry `trajectory_id` and
`trajectory_index` in their ChromaDB metadata; the trajectory record
itself lives here with its summary and keyword list.

Keywords are stored in a separate table so the canonical keyword cloud
for a project is a simple DISTINCT query, and so normalizing a keyword
touches one row rather than every EDU that mentions it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import SCHEMA_VERSION, TRAJECTORIES_DB


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectories (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    project     TEXT NOT NULL,
    start_turn  INTEGER NOT NULL,
    end_turn    INTEGER NOT NULL,
    summary     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traj_session ON trajectories(session_id);
CREATE INDEX IF NOT EXISTS idx_traj_project ON trajectories(project);
CREATE INDEX IF NOT EXISTS idx_traj_project_created ON trajectories(project, created_at DESC);

CREATE TABLE IF NOT EXISTS trajectory_keywords (
    trajectory_id  TEXT NOT NULL,
    keyword        TEXT NOT NULL,
    PRIMARY KEY (trajectory_id, keyword),
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kw_keyword ON trajectory_keywords(keyword);
"""


class SchemaVersionMismatch(RuntimeError):
    """Raised when the trajectory DB was written by a different format version."""


@dataclass
class Trajectory:
    id: str
    session_id: str
    project: str
    start_turn: int
    end_turn: int
    summary: str
    created_at: datetime
    keywords: list[str] = field(default_factory=list)


class TrajectoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else TRAJECTORIES_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            stored = int(row[0]) if row else None
            if stored is None:
                # Fresh DB — tag it with current version
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif stored != SCHEMA_VERSION:
                raise SchemaVersionMismatch(
                    f"Trajectory DB schema version is {stored}, code expects "
                    f"{SCHEMA_VERSION}. Run `claude-memory reset --all` to clean state."
                )

    def add_trajectory(self, t: Trajectory) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO trajectories "
                "(id, session_id, project, start_turn, end_turn, summary, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (t.id, t.session_id, t.project, t.start_turn, t.end_turn,
                 t.summary, t.created_at.isoformat()),
            )
            conn.execute("DELETE FROM trajectory_keywords WHERE trajectory_id = ?", (t.id,))
            if t.keywords:
                conn.executemany(
                    "INSERT OR IGNORE INTO trajectory_keywords (trajectory_id, keyword) VALUES (?, ?)",
                    [(t.id, kw) for kw in t.keywords],
                )

    def add_trajectories(self, trajectories: list[Trajectory]) -> int:
        for t in trajectories:
            self.add_trajectory(t)
        return len(trajectories)

    def get_by_id(self, traj_id: str) -> Trajectory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, session_id, project, start_turn, end_turn, summary, created_at "
                "FROM trajectories WHERE id = ?",
                (traj_id,),
            ).fetchone()
            if not row:
                return None
            keywords = [
                r[0] for r in conn.execute(
                    "SELECT keyword FROM trajectory_keywords WHERE trajectory_id = ? ORDER BY keyword",
                    (traj_id,),
                ).fetchall()
            ]
            return self._row_to_trajectory(row, keywords)

    def get_many_by_ids(self, traj_ids: list[str]) -> list[Trajectory]:
        if not traj_ids:
            return []
        placeholders = ",".join("?" * len(traj_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, session_id, project, start_turn, end_turn, summary, created_at "
                f"FROM trajectories WHERE id IN ({placeholders})",
                traj_ids,
            ).fetchall()
            kw_rows = conn.execute(
                f"SELECT trajectory_id, keyword FROM trajectory_keywords "
                f"WHERE trajectory_id IN ({placeholders}) ORDER BY keyword",
                traj_ids,
            ).fetchall()
        kw_by_traj: dict[str, list[str]] = {}
        for tid, kw in kw_rows:
            kw_by_traj.setdefault(tid, []).append(kw)
        return [self._row_to_trajectory(r, kw_by_traj.get(r[0], [])) for r in rows]

    def get_by_session(self, session_id: str) -> list[Trajectory]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, project, start_turn, end_turn, summary, created_at "
                "FROM trajectories WHERE session_id = ? ORDER BY start_turn",
                (session_id,),
            ).fetchall()
            if not rows:
                return []
            ids = [r[0] for r in rows]
        return self.get_many_by_ids(ids)

    def get_recent_by_project(self, project: str, limit: int = 10) -> list[Trajectory]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, project, start_turn, end_turn, summary, created_at "
                "FROM trajectories WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
            if not rows:
                return []
            ids = [r[0] for r in rows]
        ordered = self.get_many_by_ids(ids)
        by_id = {t.id: t for t in ordered}
        return [by_id[i] for i in ids if i in by_id]

    def get_all_by_project(self, project: str) -> list[Trajectory]:
        """All trajectories for a project, ordered by created_at DESC."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, project, start_turn, end_turn, summary, created_at "
                "FROM trajectories WHERE project = ? ORDER BY created_at DESC",
                (project,),
            ).fetchall()
            if not rows:
                return []
            ids = [r[0] for r in rows]
        ordered = self.get_many_by_ids(ids)
        by_id = {t.id: t for t in ordered}
        return [by_id[i] for i in ids if i in by_id]

    def list_projects(self) -> list[str]:
        """Return all distinct project names with at least one trajectory."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT project FROM trajectories ORDER BY project"
            ).fetchall()
        return [r[0] for r in rows]

    def get_keywords_for_project(self, project: str) -> list[str]:
        """Canonical keyword cloud for a project, alphabetized."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT kw.keyword FROM trajectory_keywords kw "
                "JOIN trajectories t ON kw.trajectory_id = t.id "
                "WHERE t.project = ? ORDER BY kw.keyword",
                (project,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_keyword_frequencies(self, project: str) -> dict[str, int]:
        """Return a map of keyword → number of distinct trajectories that mention it.

        Drives the Key Decisions/Gotchas ranking in the project index: a topic
        that recurs across many trajectories indicates an idea the user returns
        to, which correlates with importance.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kw.keyword, COUNT(DISTINCT kw.trajectory_id) "
                "FROM trajectory_keywords kw "
                "JOIN trajectories t ON kw.trajectory_id = t.id "
                "WHERE t.project = ? "
                "GROUP BY kw.keyword",
                (project,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def search_by_keywords(self, project: str, keywords: list[str]) -> list[str]:
        """Return trajectory_ids whose keyword set intersects with the given keywords,
        scoped to a single project (hard filter)."""
        if not keywords:
            return []
        placeholders = ",".join("?" * len(keywords))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT kw.trajectory_id FROM trajectory_keywords kw "
                f"JOIN trajectories t ON kw.trajectory_id = t.id "
                f"WHERE t.project = ? AND kw.keyword IN ({placeholders})",
                (project, *keywords),
            ).fetchall()
        return [r[0] for r in rows]

    def search_by_keywords_global(self, keywords: list[str]) -> list[tuple[str, str]]:
        """Cross-project keyword search. Returns list of (trajectory_id, project) tuples.

        Project is returned alongside so the caller can apply a soft boost during
        ranking without re-querying SQLite. Use this for global recall; use
        `search_by_keywords(project, ...)` when project isolation is required.
        """
        if not keywords:
            return []
        placeholders = ",".join("?" * len(keywords))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT kw.trajectory_id, t.project FROM trajectory_keywords kw "
                f"JOIN trajectories t ON kw.trajectory_id = t.id "
                f"WHERE kw.keyword IN ({placeholders})",
                tuple(keywords),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def delete_session(self, session_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM trajectories WHERE session_id = ?", (session_id,))
            return cur.rowcount

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]

    @staticmethod
    def _row_to_trajectory(row: tuple, keywords: list[str]) -> Trajectory:
        return Trajectory(
            id=row[0],
            session_id=row[1],
            project=row[2],
            start_turn=row[3],
            end_turn=row[4],
            summary=row[5],
            created_at=datetime.fromisoformat(row[6]),
            keywords=keywords,
        )
