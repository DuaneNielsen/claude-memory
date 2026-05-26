"""Manage the user-level systemd unit that runs the watcher daemon.

Subcommands wired in cli.py: install / uninstall / status / start / stop /
restart / logs. All operate on the user's `~/.config/systemd/user/claude-memory.service`
unit via `systemctl --user`.

Install resolves the venv python path from `sys.executable` and bakes it into
the unit. If the plugin venv moves (e.g. plugin reinstalled to a new location),
re-run `claude-memory service install` to regenerate the unit.
"""

import os
import subprocess
import sys
from pathlib import Path

UNIT_NAME = "claude-memory.service"
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
UNIT_PATH = UNIT_DIR / UNIT_NAME

UNIT_TEMPLATE = """[Unit]
Description=claude-memory background ingestion watcher
Documentation=https://github.com/DuaneNielsen/claude-memory
After=default.target

[Service]
Type=simple
ExecStart={python} -m claude_memory.cli watcher
Restart=on-failure
RestartSec=10
# `claude` (extractor subprocess) lives on the user's PATH — systemd user
# units don't inherit it, so re-export a sensible search list.
Environment=PATH=%h/.local/bin:%h/.npm-global/bin:%h/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
StandardOutput=journal
StandardError=journal
SyslogIdentifier=claude-memory

[Install]
WantedBy=default.target
"""


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True, text=True, check=check,
    )


def _check_systemd() -> str | None:
    """Return None if systemd --user is available, else an error string."""
    if sys.platform != "linux":
        return f"systemd-based service install is Linux-only (this is {sys.platform})."
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return "`systemctl` not found on PATH — is systemd installed?"
    # is-system-running can return non-zero ("degraded", "starting") and still
    # be perfectly fine for user services. We only care that the call works.
    if result.returncode > 4:
        return f"systemctl --user failed: {result.stderr.strip()}"
    return None


def install() -> int:
    err = _check_systemd()
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    python = sys.executable
    if not Path(python).exists():
        print(f"Error: sys.executable {python!r} does not exist", file=sys.stderr)
        return 1

    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    unit_text = UNIT_TEMPLATE.format(python=python)
    UNIT_PATH.write_text(unit_text)
    print(f"Wrote unit: {UNIT_PATH}")

    _systemctl("daemon-reload")
    _systemctl("enable", "--now", UNIT_NAME)
    print(f"Enabled and started {UNIT_NAME}.")
    print("  Tail logs:    claude-memory service logs")
    print("  Show status:  claude-memory service status")
    return 0


def uninstall() -> int:
    if not UNIT_PATH.exists():
        print(f"Unit not installed (no file at {UNIT_PATH}).")
        return 0
    # Best-effort stop + disable — don't crash if the unit is already gone.
    _systemctl("disable", "--now", UNIT_NAME, check=False)
    UNIT_PATH.unlink()
    _systemctl("daemon-reload")
    print(f"Removed {UNIT_PATH} and reloaded systemd.")
    return 0


def status() -> int:
    result = _systemctl("status", UNIT_NAME, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def start() -> int:
    result = _systemctl("start", UNIT_NAME, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def stop() -> int:
    result = _systemctl("stop", UNIT_NAME, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def restart() -> int:
    result = _systemctl("restart", UNIT_NAME, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def logs() -> int:
    """Tail-follow journald output for the watcher service."""
    try:
        os.execvp("journalctl", [
            "journalctl", "--user", "-u", UNIT_NAME, "-f", "--output=short",
        ])
    except FileNotFoundError:
        print("journalctl not found.", file=sys.stderr)
        return 1


def is_active() -> bool:
    """Returns True iff `systemctl --user is-active <unit>` exits 0.

    Used by memory_status to surface service health alongside ingestion state.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", UNIT_NAME],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0
