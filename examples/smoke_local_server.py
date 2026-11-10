"""Smoke-check every tool of a local TextLite MCP server.

Start a server first (or use --server inprocess / stdio):
    textlite-mcp --transport http --host 127.0.0.1 --port 8000
    # or: docker compose up --build -d

Then:
    python examples/smoke_local_server.py                         # http://127.0.0.1:8000/mcp
    python examples/smoke_local_server.py --server inprocess
    python examples/smoke_local_server.py --server http://127.0.0.1:8000/mcp --skip rerank_passages

The first NLP call downloads models from Hugging Face, so it can take a while.
Exits with status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from textlite_client import TextLiteClient  # noqa: E402

PARAGRAPHS = [
    "Solar panels convert sunlight into electricity using photovoltaic cells.",
    "Photovoltaic cells in solar panels turn sunlight into electric power.",
    "Wind turbines generate electricity from moving air.",
    "Battery storage lets homes use solar energy after sunset.",
]


async def run_checks(tl: TextLiteClient, skip: set[str] = frozenset()) -> list[tuple[str, bool, float, str]]:
    checks = {
        "server_info": lambda: tl.server_info(),
        "deduplicate_texts": lambda: tl.deduplicate(PARAGRAPHS, similarity_threshold=0.85),
        "cluster_entities": lambda: tl.cluster_entities(["solar panel", "solar panels", "wind turbine", "battery"]),
        "compress_contexts": lambda: tl.compress_contexts(PARAGRAPHS, num_tokens=30),
        "compress_query_contexts": lambda: tl.compress_query_contexts("How is solar power stored?", PARAGRAPHS, 30),
        "semantic_similarity": lambda: tl.semantic_similarity(PARAGRAPHS[0], PARAGRAPHS[1]),
        "semantic_search": lambda: tl.semantic_search("wind energy", PARAGRAPHS, top_k=2),
        "summarize_text": lambda: tl.summarize_text(" ".join(PARAGRAPHS), max_sentences=2),
        "summarize_paragraphs": lambda: tl.summarize_paragraphs(PARAGRAPHS, max_sentences_per_paragraph=1),
        "rerank_passages": lambda: tl.rerank("battery storage", PARAGRAPHS, top_k=2),
        "extract_keyphrases": lambda: tl.extract_keyphrases(PARAGRAPHS, top_k=8),
        "extract_topics": lambda: tl.extract_topics(PARAGRAPHS, top_k_keyphrases=20, min_topic_size=1),
    }
    results = []
    for name, check in checks.items():
        if name in skip:
            continue
        started = time.perf_counter()
        try:
            value = await check()
            ok = value is not None and value != [] and value != ""
            detail = repr(value)
        except Exception as exc:  # report and continue with the next tool
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((name, ok, time.perf_counter() - started, detail))
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http", help="inprocess | stdio | http | full MCP URL (default: http)")
    parser.add_argument("--skip", nargs="*", default=[], help="tool names to skip")
    args = parser.parse_args()

    async with TextLiteClient(args.server) as tl:
        tools = await tl.list_tools()
        print(f"Connected to {tl.info.get('server')} {tl.info.get('version')} ({len(tools)} tools)")
        print("Tools:", ", ".join(tools))
        results = await run_checks(tl, set(args.skip))

    width = max(len(name) for name, *_ in results)
    for name, ok, elapsed, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {name:<{width}}  {elapsed:6.2f}s  {detail[:120]}")
    failed = [name for name, ok, *_ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
