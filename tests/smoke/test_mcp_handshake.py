"""Layer 3 MCP smoke test: spawn the plugin's MCP server over stdio, complete
the JSON-RPC initialize handshake, and assert the expected tools register.

Does NOT exercise tool handlers — those touch ChromaDB and a sentence-transformer
model download. Tool *registration* is what proves the server-side wiring works
on a fresh deploy; tool *behavior* is covered by unit tests if/when they exist.
"""

from __future__ import annotations

import os
import sys

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

EXPECTED_TOOLS = {"memory_status", "recall_get_context"}


@pytest.mark.asyncio
async def test_server_lists_expected_tools(tmp_path):
    # Hermetic env so the server can't see the dev box's ~/.claude data.
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env.pop("CLAUDE_PLUGIN_ROOT", None)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "claude_memory"],
        env=env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            names = {t.name for t in result.tools}

    missing = EXPECTED_TOOLS - names
    assert not missing, f"server did not register expected tools: missing={missing}, got={names}"
