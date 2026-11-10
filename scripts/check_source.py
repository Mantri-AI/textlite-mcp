"""Dependency-free source/metadata smoke check for environments without ML packages."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOOLS = {
    "server_info", "deduplicate_texts", "cluster_entities", "compress_contexts",
    "compress_query_contexts", "semantic_similarity", "semantic_search",
    "summarize_text", "summarize_paragraphs", "rerank_passages",
    "extract_keyphrases", "keyphrases_to_tags", "extract_topics",
    "assign_entities_to_topics",
}


def main() -> None:
    parsed = ast.parse((ROOT / "src/textlite_mcp/server.py").read_text(encoding="utf-8"))
    exported = {
        node.name for node in parsed.body if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Attribute) and d.attr == "tool" for d in node.decorator_list)
    }
    assert REQUIRED_TOOLS <= exported, REQUIRED_TOOLS - exported
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "textlite-mcp-server"
    for path in [ROOT / "server.json.example", ROOT / "examples/claude_desktop_config.json"]:
        json.loads(path.read_text(encoding="utf-8"))
    print(f"OK: {len(exported)} tool declarations, project metadata, and example JSON")


if __name__ == "__main__":
    main()
