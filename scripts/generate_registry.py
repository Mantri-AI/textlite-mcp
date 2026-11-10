"""Create a server.json for remote-only MCP registry publishing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-user", required=True, help="Your exact GitHub username (not a display name)")
    parser.add_argument("--url", required=True, help="Public HTTPS MCP endpoint ending in /mcp")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--output", default="server.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", args.github_user):
        parser.error("--github-user must look like a GitHub username")
    parsed = urlsplit(args.url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.path.rstrip("/").endswith("/mcp"):
        parser.error("--url must be a public HTTPS URL ending in /mcp")
    doc = {
        "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
        "name": f"io.github.{args.github_user.lower()}/textlite-mcp",
        "title": "TextLite MCP",
        "description": "TextLite semantic NLP tools: compression, clustering, deduplication, summarization, similarity, keyphrases and topics",
        "version": args.version,
        "remotes": [{"type": "streamable-http", "url": args.url}],
    }
    Path(args.output).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}. Review it before publishing; no registry submission was performed.")


if __name__ == "__main__":
    main()
