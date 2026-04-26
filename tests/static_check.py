#!/usr/bin/env python3
"""Layer 1 static validator for the claude-memory plugin manifests.

Stdlib only. Run from repo root:

    python3 tests/static_check.py

Exits 0 on success, 1 on the first failure with a clear message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class CheckFail(Exception):
    pass


def _load(path: Path) -> dict:
    if not path.is_file():
        raise CheckFail(f"missing file: {path.relative_to(REPO)}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CheckFail(f"{path.relative_to(REPO)}: invalid JSON ({e})")


def _require(d: dict, key: str, where: str, kind: type | tuple[type, ...] = object) -> object:
    if key not in d:
        raise CheckFail(f"{where}: missing required key `{key}`")
    val = d[key]
    if kind is not object and not isinstance(val, kind):
        names = kind.__name__ if isinstance(kind, type) else "/".join(t.__name__ for t in kind)
        raise CheckFail(f"{where}: `{key}` must be {names}, got {type(val).__name__}")
    return val


def check_plugin_json() -> None:
    p = REPO / ".claude-plugin" / "plugin.json"
    d = _load(p)
    where = ".claude-plugin/plugin.json"
    _require(d, "name", where, str)
    _require(d, "version", where, str)
    _require(d, "description", where, str)
    mcp = _require(d, "mcpServers", where, dict)
    if not mcp:
        raise CheckFail(f"{where}: `mcpServers` is empty — plugin would expose no tools")
    for sname, scfg in mcp.items():
        sw = f"{where} mcpServers.{sname}"
        if not isinstance(scfg, dict):
            raise CheckFail(f"{sw}: must be an object")
        _require(scfg, "command", sw, str)
        _require(scfg, "args", sw, list)


def check_hooks_json() -> None:
    p = REPO / "hooks" / "hooks.json"
    d = _load(p)
    where = "hooks/hooks.json"
    hooks = _require(d, "hooks", where, dict)
    valid_events = {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SubagentStop",
        "Notification",
        "PreCompact",
    }
    for event, entries in hooks.items():
        if event not in valid_events:
            raise CheckFail(f"{where}: unknown hook event `{event}`")
        if not isinstance(entries, list):
            raise CheckFail(f"{where}: hooks.{event} must be a list")
        for i, entry in enumerate(entries):
            ew = f"{where} hooks.{event}[{i}]"
            inner = _require(entry, "hooks", ew, list)
            for j, h in enumerate(inner):
                hw = f"{ew}.hooks[{j}]"
                _require(h, "type", hw, str)
                _require(h, "command", hw, str)


def check_marketplace_json() -> None:
    p = REPO / ".claude-plugin" / "marketplace.json"
    if not p.is_file():
        return  # marketplace.json is optional
    d = _load(p)
    where = ".claude-plugin/marketplace.json"
    _require(d, "name", where, str)
    plugins = _require(d, "plugins", where, list)
    for i, entry in enumerate(plugins):
        ew = f"{where} plugins[{i}]"
        _require(entry, "name", ew, str)
        _require(entry, "version", ew, str)
        _require(entry, "source", ew, str)


def check_hook_scripts_executable() -> None:
    """Every `command` in hooks.json that points at a file in the repo must be executable."""
    hooks = json.loads((REPO / "hooks" / "hooks.json").read_text())["hooks"]
    placeholder = "${CLAUDE_PLUGIN_ROOT}"
    for event, entries in hooks.items():
        for i, entry in enumerate(entries):
            for j, h in enumerate(entry.get("hooks", [])):
                cmd = h.get("command", "")
                # Look for "${CLAUDE_PLUGIN_ROOT}/<path>" — the first whitespace-or-quote
                # delimited token containing the placeholder is the script path.
                if placeholder not in cmd:
                    continue
                # Naive but sufficient extraction: take the substring starting at the
                # placeholder and ending at the next whitespace or quote.
                start = cmd.index(placeholder)
                tail = cmd[start:]
                end = len(tail)
                for term in (' ', '"', "'"):
                    if term in tail:
                        end = min(end, tail.index(term))
                token = tail[:end]
                rel = token.replace(placeholder, "").lstrip("/")
                local = REPO / rel
                if not local.is_file():
                    continue  # could be a shell builtin path or generated file
                if not (local.stat().st_mode & 0o111):
                    raise CheckFail(
                        f"hooks.{event}[{i}].hooks[{j}]: {rel} exists but is not executable"
                    )


def main() -> int:
    checks = (
        ("plugin.json", check_plugin_json),
        ("hooks.json", check_hooks_json),
        ("marketplace.json", check_marketplace_json),
        ("hook script executability", check_hook_scripts_executable),
    )
    for name, fn in checks:
        try:
            fn()
        except CheckFail as e:
            print(f"FAIL [{name}] {e}", file=sys.stderr)
            return 1
        print(f"ok   [{name}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
