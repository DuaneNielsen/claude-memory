#!/usr/bin/env bash
# Re-run layers 1-3 inside a fresh-distro container. Invoked by Dockerfiles.
#
# Layer 1: static manifest validation (stdlib python, no plugin deps required).
# Layer 3: MCP handshake + hook smoke tests.
#
# Layer 2 (unit tests) is skipped here — there are no unit tests yet. When
# tests/unit/ exists, add another pytest invocation between the two below.

set -euo pipefail

cd /plugin

echo "::group::layer-1 static validation"
python3 tests/static_check.py
echo "::endgroup::"

echo "::group::layer-3 MCP + hook smoke"
.venv/bin/pytest tests/smoke -v
echo "::endgroup::"

echo "deploy probe OK"
