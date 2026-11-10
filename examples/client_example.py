"""Example MCP client; use in-memory server or remote HTTP endpoint.

Run after installation:
    python examples/client_example.py
    python examples/client_example.py https://YOUR_HOST/mcp
"""

import asyncio
import sys

from fastmcp import Client
from textlite_mcp.server import mcp


async def main() -> None:
    server = sys.argv[1] if len(sys.argv) > 1 else mcp
    async with Client(server) as client:
        tools = await client.list_tools()
        print("Available tools:", ", ".join(tool.name for tool in tools))
        info = await client.call_tool("server_info", {})
        print("Server info:", info.data)
        response = await client.call_tool(
            "deduplicate_texts",
            {"texts": ["Python is popular.", "Python is popular.", "Python is used in AI."]},
        )
        print("Deduplicated:", response.data)


if __name__ == "__main__":
    asyncio.run(main())
