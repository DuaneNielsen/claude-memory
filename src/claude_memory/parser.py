"""Parse Claude Code JSONL session files into conversation turns."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import CLAUDE_PROJECTS_DIR


@dataclass
class Turn:
    turn_id: int
    session_id: str
    project: str
    timestamp: datetime
    speaker: str  # "user" or "assistant"
    text: str
    git_branch: str | None = None


@dataclass
class Session:
    session_id: str
    project: str
    file_path: Path
    turns: list[Turn] = field(default_factory=list)
    line_count: int = 0
    file_hash: str | None = None

    @property
    def timestamp(self) -> datetime | None:
        return self.turns[0].timestamp if self.turns else None


def extract_text(content) -> str:
    """Extract text from message content (string or list of content blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return ""


def parse_session_file(file_path: Path) -> Session | None:
    """Parse a single JSONL session file into a Session with turns."""
    session_id = file_path.stem
    project = file_path.parent.name  # e.g. "-home-duane-primesignal"
    # Clean up project name: "-home-duane-primesignal" -> "primesignal" (or "home" for "-home-duane")
    project = _clean_project_name(project)

    turns: list[Turn] = []
    turn_id = 0
    line_count = 0

    with open(file_path) as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            message = msg.get("message", {})
            content = message.get("content")
            if content is None:
                continue

            text = extract_text(content)
            if not text:
                continue

            timestamp = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            git_branch = msg.get("gitBranch")

            turns.append(Turn(
                turn_id=turn_id,
                session_id=session_id,
                project=project,
                timestamp=timestamp,
                speaker=msg_type,
                text=text,
                git_branch=git_branch,
            ))
            turn_id += 1

    if not turns:
        return None

    return Session(
        session_id=session_id,
        project=project,
        file_path=file_path,
        turns=turns,
        line_count=line_count,
    )


def _clean_project_name(raw: str) -> str:
    """Convert directory name like '-home-duane-primesignal' to 'primesignal'."""
    parts = raw.strip("-").split("-")
    # Skip "home" and username prefix: "-home-duane-primesignal" -> "primesignal"
    # "-home-duane" alone -> "home" (the home directory project)
    if len(parts) >= 2 and parts[0] == "home":
        remainder = "-".join(parts[2:]) if len(parts) > 2 else "home"
        return remainder or "home"
    return raw


def project_from_cwd(cwd: Path | str | None = None) -> str:
    """Derive the project name used by ingestion from a filesystem path.

    Mirrors the encoding Claude Code uses for session storage directories:
    /home/duane/projects/claude-memory -> -home-duane-projects-claude-memory
    -> projects-claude-memory (after _clean_project_name).
    """
    path = Path(cwd) if cwd else Path.cwd()
    encoded = "-" + str(path.resolve()).lstrip("/").replace("/", "-")
    return _clean_project_name(encoded)


def discover_sessions(projects_dir: Path | None = None) -> list[Path]:
    """Find all JSONL session files."""
    base = projects_dir or CLAUDE_PROJECTS_DIR
    return sorted(base.glob("*/*.jsonl"))


def parse_all_sessions(projects_dir: Path | None = None) -> list[Session]:
    """Parse all session files and return sessions with turns."""
    sessions = []
    for path in discover_sessions(projects_dir):
        session = parse_session_file(path)
        if session:
            sessions.append(session)
    return sessions
