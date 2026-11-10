"""Text-compression examples against a local TextLite MCP server.

Uses ``test_compression.txt`` (next to this script) unless --file is given.

    python examples/compression_example.py                                  # in-process server
    python examples/compression_example.py --server http                    # http://127.0.0.1:8000/mcp
    python examples/compression_example.py --file notes.txt --budgets 100 300 \\
        --query "How does battery storage help solar power?"

Examples:
  1. Generic compression at several token budgets (compress_contexts).
  2. Query-focused compression (compress_query_contexts).
  3. Extractive summary of the whole text (summarize_text / summarize_paragraphs).
  4. Near-duplicate paragraphs removed before compression (deduplicate_texts).
  5. Keyphrase coverage: which original keyphrases survive compression.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from textlite_client import (  # noqa: E402
    TextLiteClient,
    count_tokens,
    format_compression_report,
    read_text_file,
    split_paragraphs,
)

DEFAULT_FILE = Path(__file__).resolve().parent / "test_compression.txt"
DEFAULT_QUERY = "How does battery storage help solar power?"


def heading(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


async def example_budgets(tl: TextLiteClient, text: str, source: str, budgets: list[int]) -> None:
    heading("1. Generic compression at several budgets")
    results = await tl.compare_budgets(text, budgets, source=source)
    print(f"{'budget':>8} {'original':>9} {'compressed':>11} {'contexts':>9} {'smaller':>8} {'time':>7}")
    for result in results:
        row = result.summary_row()
        print(
            f"{row['budget_tokens']:>8} {row['original_tokens']:>9} {row['compressed_tokens']:>11} "
            f"{row['contexts']:>9} {row['reduction_pct']:>7}% {row['elapsed_s']:>6}s"
        )
    print()
    print(format_compression_report(results[0]))


async def example_query(tl: TextLiteClient, text: str, source: str, query: str, budget: int) -> None:
    heading("2. Query-focused compression")
    result = await tl.compress_text(text, budget, query=query, source=source, min_similarity=0.2)
    print(format_compression_report(result))


async def example_summary(tl: TextLiteClient, text: str) -> None:
    heading("3. Extractive summaries")
    paragraphs = split_paragraphs(text)
    if len(text) <= tl.limits["max_chars_per_text"]:
        print("Whole text (5 sentences):")
        print(await tl.summarize_text(text, max_sentences=5))
    else:
        print("Text exceeds the server's per-text limit; summarizing per paragraph instead.")
    print("\nFirst paragraphs (1 sentence each):")
    for summary in (await tl.summarize_paragraphs(paragraphs[:5], max_sentences_per_paragraph=1)):
        print(f"- {summary}")


async def example_dedup(tl: TextLiteClient, text: str, budget: int) -> None:
    heading("4. Deduplicate paragraphs, then compress")
    paragraphs = split_paragraphs(text)[: tl.limits["max_texts"]]
    dedup = await tl.deduplicate(paragraphs, similarity_threshold=0.85)
    duplicates = [group for group in dedup["groups"] if len(group) > 1]
    print(f"{len(paragraphs)} paragraphs -> {len(dedup['unique_texts'])} unique")
    for group in duplicates:
        print(f"  duplicate group (paragraph indices): {group}")
    contexts = await tl.compress_contexts(dedup["unique_texts"], num_tokens=budget)
    print(f"\nCompressed unique paragraphs to {count_tokens(' '.join(contexts))} tokens:")
    for context in contexts:
        print(f"- {context}")


async def example_keyphrase_coverage(tl: TextLiteClient, text: str, budget: int) -> None:
    heading("5. Keyphrase coverage after compression")
    paragraphs = split_paragraphs(text)[: tl.limits["max_texts"]]
    keyphrases = await tl.extract_keyphrases(paragraphs, top_k=15, max_ngram=3)
    compressed = (await tl.compress_text(text, budget)).compressed_text.lower()
    kept = [k["text"] for k in keyphrases if k["text"].lower() in compressed]
    lost = [k["text"] for k in keyphrases if k["text"].lower() not in compressed]
    print(f"Budget {budget}: {len(kept)}/{len(keyphrases)} top keyphrases kept")
    print("kept:", ", ".join(kept) or "-")
    print("lost:", ", ".join(lost) or "-")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="inprocess", help="inprocess | stdio | http | full MCP URL")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="text file to compress")
    parser.add_argument("--budgets", type=int, nargs="+", default=[80, 150, 300], help="token budgets")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="query for query-focused compression")
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"{args.file} not found")
    text = read_text_file(args.file)
    print(f"{args.file}: {len(text)} chars, ~{count_tokens(text)} tokens, {len(split_paragraphs(text))} paragraphs")

    async with TextLiteClient(args.server) as tl:
        budget = args.budgets[0]
        await example_budgets(tl, text, args.file.name, args.budgets)
        await example_query(tl, text, args.file.name, args.query, budget)
        await example_summary(tl, text)
        await example_dedup(tl, text, budget)
        await example_keyphrase_coverage(tl, text, budget)


if __name__ == "__main__":
    asyncio.run(main())
