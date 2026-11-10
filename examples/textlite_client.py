"""Reusable helpers for exercising a local TextLite MCP server.

``TextLiteClient(server)`` accepts:

    "inprocess"                  the server object imported from ``textlite_mcp`` (no process)
    "stdio"                      spawns ``python -m textlite_mcp.server --transport stdio``
    "http"                       ``$TEXTLITE_MCP_URL`` or http://127.0.0.1:8000/mcp
    "http://HOST:PORT/mcp"       any running HTTP server (``textlite-mcp --transport http``
                                 or ``docker compose up``)

Script usage::

    import asyncio
    from textlite_client import TextLiteClient, format_compression_report

    async def main():
        async with TextLiteClient("http") as tl:
            result = await tl.compress_file("test_compression.txt", num_tokens=200)
            print(format_compression_report(result))

    asyncio.run(main())

In Jupyter, ``await`` the coroutines directly: the kernel already runs an event loop.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from fastmcp import Client

DEFAULT_HTTP_URL = os.getenv("TEXTLITE_MCP_URL", "http://127.0.0.1:8000/mcp")

# Server defaults (see textlite_mcp/server.py); replaced by server_info() on connect.
DEFAULT_LIMITS = {"max_texts": 100, "max_chars_per_text": 20000, "max_total_chars": 120000}

# Server-side cap on num_tokens / max_tokens (see _positive in textlite_mcp/server.py).
MAX_TOKEN_BUDGET = 20000

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

_GENERIC_COMPRESSION_OPTIONS = {
    "dedup_threshold",
    "similarity_budget_threshold",
    "paragraph_summary_max_sentences",
    "long_paragraph_token_threshold",
}
_QUERY_COMPRESSION_OPTIONS = {"dedup_threshold", "min_similarity"}


# ---- Text helpers -------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Approximate token count; same rule as TextLite's ``cheap_token_count``."""
    return len(_TOKEN_RE.findall(text or ""))


def read_text_file(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def split_paragraphs(text: str, max_chars: int = 1500) -> list[str]:
    """Blank-line separated paragraphs with whitespace collapsed.

    Paragraphs longer than ``max_chars`` (e.g. a transcript without blank lines)
    are split on sentence boundaries into chunks of at most ``max_chars``.
    """
    paragraphs: list[str] = []
    for block in _PARAGRAPH_SPLIT_RE.split(text or ""):
        block = " ".join(block.split())
        if not block:
            continue
        if len(block) <= max_chars:
            paragraphs.append(block)
            continue
        chunk = ""
        for sentence in _SENTENCE_SPLIT_RE.split(block):
            sentence = sentence.strip()
            while len(sentence) > max_chars:
                if chunk:
                    paragraphs.append(chunk)
                    chunk = ""
                paragraphs.append(sentence[:max_chars])
                sentence = sentence[max_chars:].strip()
            if not sentence:
                continue
            if chunk and len(chunk) + 1 + len(sentence) > max_chars:
                paragraphs.append(chunk)
                chunk = sentence
            else:
                chunk = f"{chunk} {sentence}".strip()
        if chunk:
            paragraphs.append(chunk)
    return paragraphs


def batch_paragraphs(paragraphs: Sequence[str], limits: dict[str, int] | None = None) -> list[list[str]]:
    """Group paragraphs into consecutive batches that respect the server's input limits."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for paragraph in paragraphs:
        paragraph = paragraph[: limits["max_chars_per_text"]]
        if current and (
            len(current) >= limits["max_texts"]
            or current_chars + len(paragraph) > limits["max_total_chars"]
        ):
            batches.append(current)
            current, current_chars = [], 0
        current.append(paragraph)
        current_chars += len(paragraph)
    if current:
        batches.append(current)
    return batches


# ---- Results ------------------------------------------------------------------------
@dataclass
class CompressionResult:
    contexts: list[str]
    num_tokens: int
    query: str | None
    original_chars: int
    original_tokens: int
    input_paragraphs: int
    elapsed_s: float
    rounds: int = 1
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def compressed_text(self) -> str:
        return "\n\n".join(self.contexts)

    @property
    def compressed_tokens(self) -> int:
        return count_tokens(self.compressed_text)

    @property
    def ratio(self) -> float:
        """Compressed / original tokens (lower means more compression)."""
        return self.compressed_tokens / self.original_tokens if self.original_tokens else 0.0

    @property
    def reduction_pct(self) -> float:
        return (1.0 - self.ratio) * 100 if self.original_tokens else 0.0

    def summary_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "query": self.query or "",
            "budget_tokens": self.num_tokens,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "contexts": len(self.contexts),
            "reduction_pct": round(self.reduction_pct, 1),
            "rounds": self.rounds,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def format_compression_report(result: CompressionResult, preview_chars: int = 800) -> str:
    title = f"Compression of {result.source or 'text'}"
    if result.query:
        title += f" for query {result.query!r}"
    lines = [
        title,
        "-" * len(title),
        f"budget: {result.num_tokens} tokens | rounds: {result.rounds} | time: {result.elapsed_s:.2f}s",
        f"original:   {result.original_tokens} tokens, {result.original_chars} chars, "
        f"{result.input_paragraphs} paragraphs",
        f"compressed: {result.compressed_tokens} tokens in {len(result.contexts)} contexts "
        f"({result.reduction_pct:.1f}% smaller)",
    ]
    if result.compressed_tokens > result.num_tokens:
        lines.append("note: over budget (TextLite may keep one oversized context; counts are approximate)")
    preview = result.compressed_text
    if preview_chars and len(preview) > preview_chars:
        preview = preview[:preview_chars].rstrip() + " ..."
    lines += ["", preview]
    return "\n".join(lines)


# ---- Client -------------------------------------------------------------------------
def resolve_server(server: Any = "inprocess") -> Any:
    """Turn a server shorthand into something ``fastmcp.Client`` accepts."""
    if server in (None, "inprocess"):
        from textlite_mcp.server import mcp

        return mcp
    if server == "stdio":
        from fastmcp.client.transports import StdioTransport

        return StdioTransport(
            command=sys.executable,
            args=["-m", "textlite_mcp.server", "--transport", "stdio"],
            env=dict(os.environ),
        )
    if server == "http":
        return DEFAULT_HTTP_URL
    return server


def _tool_data(result: Any) -> Any:
    """Python value of a tool result across fastmcp result shapes."""
    data = getattr(result, "data", None)
    if data is not None:
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured.get("result", structured)
    content = getattr(result, "content", None) or []
    return "\n".join(getattr(item, "text", "") for item in content)


class TextLiteClient:
    """Async wrapper with one method per TextLite MCP tool plus file/text compression."""

    def __init__(self, server: Any = "inprocess", *, timeout: float | None = 600):
        self.server = server
        self.timeout = timeout
        self.limits = dict(DEFAULT_LIMITS)
        self.info: dict[str, Any] = {}
        self._client: Client | None = None

    async def __aenter__(self) -> "TextLiteClient":
        client = Client(resolve_server(self.server), timeout=self.timeout)
        await client.__aenter__()
        self._client = client
        try:
            self.info = await self.server_info()
            self.limits.update(self.info.get("limits") or {})
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.__aexit__(*exc_info)

    async def call(self, tool: str, **arguments: Any) -> Any:
        """Call any tool; ``None`` arguments are omitted so server defaults apply."""
        if self._client is None:
            raise RuntimeError("Use 'async with TextLiteClient(...) as tl:' before calling tools")
        payload = {key: value for key, value in arguments.items() if value is not None}
        return _tool_data(await self._client.call_tool(tool, payload))

    # -- Discovery
    async def list_tools(self) -> list[str]:
        if self._client is None:
            raise RuntimeError("Use 'async with TextLiteClient(...) as tl:' before calling tools")
        return sorted(tool.name for tool in await self._client.list_tools())

    async def server_info(self) -> dict[str, Any]:
        return await self.call("server_info")

    # -- Deduplication / clustering / similarity
    async def deduplicate(self, texts: Sequence[str], similarity_threshold: float = 0.9) -> dict[str, Any]:
        return await self.call("deduplicate_texts", texts=list(texts), similarity_threshold=similarity_threshold)

    async def cluster_entities(
        self, entities: Sequence[str], similarity_threshold: float = 0.72, dedup_threshold: float = 0.92
    ) -> list[dict[str, Any]]:
        return await self.call(
            "cluster_entities", entities=list(entities),
            similarity_threshold=similarity_threshold, dedup_threshold=dedup_threshold,
        )

    async def semantic_similarity(self, text_a: str, text_b: str) -> float:
        return float((await self.call("semantic_similarity", text_a=text_a, text_b=text_b))["score"])

    async def semantic_search(self, query: str, texts: Sequence[str], top_k: int = 5) -> list[dict[str, Any]]:
        return await self.call("semantic_search", query=query, texts=list(texts), top_k=top_k)

    async def rerank(self, query: str, passages: Sequence[str], top_k: int = 5) -> list[str]:
        return await self.call("rerank_passages", query=query, passages=list(passages), top_k=top_k)

    # -- Compression
    async def compress_contexts(self, paragraphs: Sequence[str], num_tokens: int = 250, **options: Any) -> list[str]:
        """Representative, diverse paragraphs under an approximate token budget."""
        return await self.call("compress_contexts", paragraphs=list(paragraphs), num_tokens=num_tokens, **options)

    async def compress_query_contexts(
        self, query: str, paragraphs: Sequence[str], num_tokens: int = 250, **options: Any
    ) -> list[str]:
        """Query-relevant sentences under an approximate token budget."""
        return await self.call(
            "compress_query_contexts", query=query, paragraphs=list(paragraphs), num_tokens=num_tokens, **options
        )

    async def _compress_batch(
        self, paragraphs: list[str], num_tokens: int, query: str | None, options: dict[str, Any]
    ) -> list[str]:
        if query:
            allowed = {k: v for k, v in options.items() if k in _QUERY_COMPRESSION_OPTIONS}
            return await self.compress_query_contexts(query, paragraphs, num_tokens, **allowed)
        allowed = {k: v for k, v in options.items() if k in _GENERIC_COMPRESSION_OPTIONS}
        return await self.compress_contexts(paragraphs, num_tokens, **allowed)

    async def compress_text(
        self,
        text: str,
        num_tokens: int = 250,
        *,
        query: str | None = None,
        paragraph_chars: int = 1500,
        max_rounds: int = 5,
        source: str = "",
        **options: Any,
    ) -> CompressionResult:
        """Compress arbitrary-length text.

        The text is split into paragraphs. If they exceed the server's input
        limits, each batch is compressed to ``num_tokens`` and the combined
        results are compressed again (map-reduce) until a single call fits.
        ``options`` are passed to the matching tool (unknown ones are dropped).
        """
        if not 1 <= num_tokens <= MAX_TOKEN_BUDGET:
            raise ValueError(f"num_tokens must be between 1 and {MAX_TOKEN_BUDGET}")
        max_chars = min(paragraph_chars, self.limits["max_chars_per_text"])
        paragraphs = split_paragraphs(text, max_chars=max_chars)
        started = time.perf_counter()
        current, rounds = paragraphs, 0
        contexts: list[str] = []
        while current:
            rounds += 1
            batches = batch_paragraphs(current, self.limits)
            if len(batches) == 1:
                contexts = await self._compress_batch(batches[0], num_tokens, query, options)
                break
            if rounds >= max_rounds:
                raise RuntimeError(f"Text still exceeds server limits after {rounds} compression rounds")
            reduced: list[str] = []
            for batch in batches:
                reduced.extend(await self._compress_batch(batch, num_tokens, query, options))
            if sum(map(len, reduced)) >= sum(map(len, current)):
                raise RuntimeError("Compression did not reduce the text; lower num_tokens")
            current = split_paragraphs("\n\n".join(reduced), max_chars=max_chars)
        return CompressionResult(
            contexts=contexts,
            num_tokens=num_tokens,
            query=query,
            original_chars=len(text),
            original_tokens=count_tokens(text),
            input_paragraphs=len(paragraphs),
            elapsed_s=time.perf_counter() - started,
            rounds=rounds,
            source=source,
        )

    async def compress_file(
        self, path: str | Path, num_tokens: int = 250, *, query: str | None = None, **options: Any
    ) -> CompressionResult:
        return await self.compress_text(
            read_text_file(path), num_tokens, query=query, source=Path(path).name, **options
        )

    async def compare_budgets(
        self, text: str, budgets: Sequence[int], *, query: str | None = None, source: str = "", **options: Any
    ) -> list[CompressionResult]:
        return [
            await self.compress_text(text, budget, query=query, source=source, **options)
            for budget in budgets
        ]

    # -- Summaries / keyphrases / topics
    async def summarize_text(
        self, text: str, max_sentences: int = 3, max_tokens: int | None = None, diversity: float = 0.1
    ) -> str:
        return await self.call(
            "summarize_text", text=text, max_sentences=max_sentences, max_tokens=max_tokens, diversity=diversity
        )

    async def summarize_paragraphs(
        self, paragraphs: Sequence[str], max_sentences_per_paragraph: int = 2, max_tokens: int | None = None
    ) -> list[str]:
        return await self.call(
            "summarize_paragraphs", paragraphs=list(paragraphs),
            max_sentences_per_paragraph=max_sentences_per_paragraph, max_tokens=max_tokens,
        )

    async def extract_keyphrases(self, texts: Sequence[str], top_k: int = 20, **options: Any) -> list[dict[str, Any]]:
        return await self.call("extract_keyphrases", texts=list(texts), top_k=top_k, **options)

    async def keyphrases_to_tags(self, keyphrases: Sequence[dict[str, Any]], source: str = "textlite_keyphrases"):
        return await self.call("keyphrases_to_tags", keyphrases=list(keyphrases), source=source)

    async def extract_topics(self, texts: Sequence[str], **options: Any) -> list[dict[str, Any]]:
        return await self.call("extract_topics", texts=list(texts), **options)

    async def assign_entities_to_topics(
        self, topics: Sequence[dict[str, Any]], entities: Sequence[dict[str, Any]], max_entities_per_topic: int = 12
    ) -> list[dict[str, Any]]:
        return await self.call(
            "assign_entities_to_topics", topics=list(topics), entities=list(entities),
            max_entities_per_topic=max_entities_per_topic,
        )
