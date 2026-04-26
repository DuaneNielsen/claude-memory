"""Layer 3 hook smoke tests.

Each hook script runs as a subprocess with a synthetic env / stdin and is
asserted on:
  * exits 0 (a crashing hook would silently break sessions)
  * stdout JSON shape, where the hook is expected to emit one

Hooks that fork detached background work (clear_ingest.sh kicks off an ingest
via nohup) are tested for the early-exit paths only — we don't want CI
spawning real ingest jobs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
HOOKS = REPO / "hooks"


def _run(cmd, *, env=None, stdin: str = "", check_exit: int = 0) -> subprocess.CompletedProcess:
    base_env = os.environ.copy()
    if env:
        base_env.update(env)
    cp = subprocess.run(
        cmd,
        input=stdin,
        capture_output=True,
        text=True,
        env=base_env,
        timeout=30,
    )
    assert cp.returncode == check_exit, (
        f"command {cmd!r} exited {cp.returncode} (expected {check_exit})\n"
        f"stdout: {cp.stdout!r}\nstderr: {cp.stderr!r}"
    )
    return cp


def test_session_start_index_silent_when_no_index(tmp_path, monkeypatch):
    # No index file exists → hook exits 0 silently (no JSON on stdout).
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_project = tmp_path / "fake-proj"
    fake_project.mkdir()
    cp = _run(
        [sys.executable, str(HOOKS / "session_start_index.py")],
        env={"CLAUDE_PROJECT_DIR": str(fake_project), "HOME": str(tmp_path)},
    )
    assert cp.stdout.strip() == "", f"expected no stdout, got: {cp.stdout!r}"


def test_session_start_index_emits_context_when_index_exists(tmp_path):
    # Drop a fake index file at the path the hook will compute from PWD.
    fake_project = tmp_path / "fake-proj"
    fake_project.mkdir()

    # Mirror the hook's _project_from_cwd derivation so we know where to drop the index.
    encoded = "-" + str(fake_project.resolve()).lstrip("/").replace("/", "-")
    parts = encoded.strip("-").split("-")
    if len(parts) >= 2 and parts[0] == "home":
        project_id = "-".join(parts[2:]) if len(parts) > 2 else "home"
    else:
        project_id = encoded

    indices_dir = tmp_path / ".local" / "share" / "claude-memory" / "indices"
    indices_dir.mkdir(parents=True)
    (indices_dir / f"{project_id}.md").write_text("# fake index\n\nsmoke-test marker\n")

    # The hook reads PWD from env before falling back to os.getcwd, so set both.
    cp = subprocess.run(
        [sys.executable, str(HOOKS / "session_start_index.py")],
        capture_output=True,
        text=True,
        cwd=str(fake_project),
        env={**os.environ, "HOME": str(tmp_path), "PWD": str(fake_project)},
        timeout=30,
    )
    assert cp.returncode == 0, f"stderr: {cp.stderr}"
    assert cp.stdout.strip(), f"hook produced no output despite index file at {project_id}.md"
    payload = json.loads(cp.stdout)
    additional = payload["hookSpecificOutput"]["additionalContext"]
    assert "smoke-test marker" in additional


def test_memory_check_reset_clears_flag(tmp_path):
    flag = Path("/tmp/claude-memory-prompted")
    flag.touch()
    _run(["bash", str(HOOKS / "memory-check-reset.sh")])
    assert not flag.exists(), "reset hook did not delete the prompted-flag file"


def test_memory_check_early_exit_when_already_prompted(tmp_path):
    flag = Path("/tmp/claude-memory-prompted")
    flag.touch()
    try:
        cp = _run(
            ["bash", str(HOOKS / "memory-check.sh")],
            env={"CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        assert cp.stdout.strip() == "", f"expected early exit with no output, got: {cp.stdout!r}"
    finally:
        flag.unlink(missing_ok=True)


def test_memory_check_silent_when_zero_pending(tmp_path):
    # Clear flag; point HOME at empty dir so get_pending_sessions sees zero.
    Path("/tmp/claude-memory-prompted").unlink(missing_ok=True)
    cp = _run(
        ["bash", str(HOOKS / "memory-check.sh")],
        env={"CLAUDE_PLUGIN_ROOT": str(REPO), "HOME": str(tmp_path)},
    )
    # 0 pending → no JSON emitted (only the >0 branches print).
    assert cp.stdout.strip() == "", f"expected no output for 0 pending, got: {cp.stdout!r}"
    # Cleanup the flag the hook just created.
    Path("/tmp/claude-memory-prompted").unlink(missing_ok=True)


def test_clear_ingest_early_exit_without_session_id(tmp_path):
    # No session_id in payload → hook exits 0 without forking any ingest.
    if shutil.which("jq") is None:
        pytest.skip("jq not available; clear_ingest.sh requires it")
    cp = _run(
        ["bash", str(HOOKS / "clear_ingest.sh")],
        env={"CLAUDE_PLUGIN_ROOT": str(REPO), "CLAUDE_PLUGIN_DATA": str(tmp_path)},
        stdin="{}",
    )
    assert cp.stdout.strip() == ""
