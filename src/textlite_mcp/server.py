"""FastMCP server wrapping the public TextLite API.

All tools accept/return JSON-compatible data and never require an LLM API key.
Text embeddings and reranking models are loaded locally on first use.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import os
from typing import Any

import numpy as np
from fastmcp import FastMCP

from textlite import (
    Keyphrase,
    Topic,
    assign_entities_to_topics as tl_assign_entities_to_topics,
    cluster_entities as tl_cluster_entities,
    compress_contexts as tl_compress_contexts,
    compress_query_contexts as tl_compress_query_contexts,
    deduplicate_texts as tl_deduplicate_texts,
    extract_keyphrases as tl_extract_keyphrases,
    extract_topics as tl_extract_topics,
    keyphrases_to_tags as tl_keyphrases_to_tags,
    rerank_passages as tl_rerank_passages,
    summarize_paragraphs as tl_summarize_paragraphs,
    summarize_text as tl_summarize_text,
)
from textlite.embeddings import SentenceTransformerEmbedder
from textlite.reranking import CrossEncoderReranker

from . import __version__

mcp = FastMCP(
    name="TextLite MCP",
    instructions=(
        "Semantic NLP tools for deduplication, clustering, context compression, "
        "summarization, keyphrases, topic extraction, reranking and search. "
        "Input sizes are capped. Models run locally and may download on first use."
    ),
)

_EMBEDDING_MODEL = os.getenv("TEXTLITE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_QUERY_MODEL = os.getenv("TEXTLITE_QUERY_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
_RERANK_MODEL = os.getenv("TEXTLITE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_MAX_TEXTS = int(os.getenv("TEXTLITE_MAX_TEXTS", "100"))
_MAX_CHARS_PER_TEXT = int(os.getenv("TEXTLITE_MAX_CHARS_PER_TEXT", "20000"))
_MAX_TOTAL_CHARS = int(os.getenv("TEXTLITE_MAX_TOTAL_CHARS", "120000"))


@lru_cache(maxsize=4)
def _embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Reuse embedding models across MCP requests in a process."""
    return SentenceTransformerEmbedder(model_name=model_name)


@lru_cache(maxsize=2)
def _reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)


def _texts(items: list[str], *, allow_empty: bool = False) -> list[str]:
    if len(items) > _MAX_TEXTS:
        raise ValueError(f"Too many texts: maximum {_MAX_TEXTS} per request")
    if any(not isinstance(item, str) for item in items):
        raise ValueError("Each item must be a string")
    if any(len(item) > _MAX_CHARS_PER_TEXT for item in items):
        raise ValueError(f"Each text must be at most {_MAX_CHARS_PER_TEXT} characters")
    if sum(map(len, items)) > _MAX_TOTAL_CHARS:
        raise ValueError(f"Combined text must be at most {_MAX_TOTAL_CHARS} characters")
    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned and not allow_empty:
        raise ValueError("Provide at least one nonempty text")
    return cleaned


def _one(text: str, label: str = "text") -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label} cannot be empty")
    if len(text) > min(_MAX_TOTAL_CHARS, _MAX_CHARS_PER_TEXT):
        raise ValueError(f"{label} is too long (max {_MAX_CHARS_PER_TEXT} characters)")
    return text.strip()


def _threshold(value: float, name: str) -> float:
    if not np.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _positive(value: int, name: str, maximum: int = 20000) -> int:
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _cosine_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Compute cosine scores even when a custom embedder does not normalize."""
    matrix = np.asarray(matrix, dtype=float)
    query_vector = np.asarray(query_vector, dtype=float)
    norm = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    return np.divide(matrix @ query_vector, norm, out=np.zeros(len(matrix)), where=norm > 0)


@mcp.tool
def server_info() -> dict[str, Any]:
    """Return server version, configured models and input limits (no models loaded)."""
    return {
        "server": "TextLite MCP",
        "version": __version__,
        "embedding_model": _EMBEDDING_MODEL,
        "query_model": _QUERY_MODEL,
        "rerank_model": _RERANK_MODEL,
        "limits": {
            "max_texts": _MAX_TEXTS,
            "max_chars_per_text": _MAX_CHARS_PER_TEXT,
            "max_total_chars": _MAX_TOTAL_CHARS,
        },
    }


@mcp.tool
def deduplicate_texts(texts: list[str], similarity_threshold: float = 0.9) -> dict[str, Any]:
    """Remove normalized-exact and semantically similar duplicates; preserve duplicate groups."""
    _threshold(similarity_threshold, "similarity_threshold")
    items = _texts(texts, allow_empty=True)
    result = tl_deduplicate_texts(items, threshold=similarity_threshold, embedder=_embedder(_EMBEDDING_MODEL))
    return asdict(result)


@mcp.tool
def cluster_entities(
    entities: list[str], similarity_threshold: float = 0.72, dedup_threshold: float = 0.92
) -> list[dict[str, Any]]:
    """Cluster related entity names or phrases; output members, label, representative and cohesion."""
    _threshold(similarity_threshold, "similarity_threshold")
    _threshold(dedup_threshold, "dedup_threshold")
    items = _texts(entities, allow_empty=True)
    return [
        asdict(cluster)
        for cluster in tl_cluster_entities(
            items, similarity_threshold=similarity_threshold,
            dedup_threshold=dedup_threshold, embedder=_embedder(_EMBEDDING_MODEL)
        )
    ]


@mcp.tool
def compress_contexts(
    paragraphs: list[str],
    num_tokens: int = 250,
    dedup_threshold: float = 0.90,
    similarity_budget_threshold: float = 0.80,
    paragraph_summary_max_sentences: int = 2,
    long_paragraph_token_threshold: int = 120,
) -> list[str]:
    """Choose informative, diverse paragraphs under an approximate text-token budget."""
    _positive(num_tokens, "num_tokens")
    _positive(paragraph_summary_max_sentences, "paragraph_summary_max_sentences", 30)
    _positive(long_paragraph_token_threshold, "long_paragraph_token_threshold")
    _threshold(dedup_threshold, "dedup_threshold")
    _threshold(similarity_budget_threshold, "similarity_budget_threshold")
    return tl_compress_contexts(
        _texts(paragraphs, allow_empty=True),
        num_tokens=num_tokens,
        dedup_threshold=dedup_threshold,
        similarity_budget_threshold=similarity_budget_threshold,
        paragraph_summary_max_sentences=paragraph_summary_max_sentences,
        long_paragraph_token_threshold=long_paragraph_token_threshold,
        embedder=_embedder(_EMBEDDING_MODEL),
    )


@mcp.tool
def compress_query_contexts(
    query: str,
    paragraphs: list[str],
    num_tokens: int = 250,
    dedup_threshold: float = 0.9,
    min_similarity: float = 0.2,
) -> list[str]:
    """Extract question-relevant sentences under an approximate text-token budget."""
    _positive(num_tokens, "num_tokens")
    _threshold(dedup_threshold, "dedup_threshold")
    # Cosine may be negative; TextLite compares directly to this lower cutoff.
    if not (-1.0 <= min_similarity <= 1.0):
        raise ValueError("min_similarity must be between -1 and 1")
    return tl_compress_query_contexts(
        _one(query, "query"),
        _texts(paragraphs, allow_empty=True),
        num_tokens,
        dedup_threshold=dedup_threshold,
        min_similarity=min_similarity,
        embedder=_embedder(_QUERY_MODEL),
    )


@mcp.tool
def semantic_similarity(text_a: str, text_b: str) -> dict[str, Any]:
    """Return the cosine similarity (-1 to 1) between two locally embedded texts."""
    left, right = _one(text_a, "text_a"), _one(text_b, "text_b")
    vectors = _embedder(_EMBEDDING_MODEL).encode([left, right])
    return {"text_a": left, "text_b": right, "score": float(_cosine_scores(vectors[1:], vectors[0])[0])}


@mcp.tool
def semantic_search(query: str, texts: list[str], top_k: int = 5) -> list[dict[str, Any]]:
    """Return top-k candidate texts ordered by cosine similarity to a query."""
    _positive(top_k, "top_k", _MAX_TEXTS)
    q = _one(query, "query")
    candidates = _texts(texts, allow_empty=True)
    if not candidates:
        return []
    vectors = _embedder(_QUERY_MODEL).encode([q, *candidates])
    scores = _cosine_scores(vectors[1:], vectors[0])
    ranked = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))[:top_k]
    return [{"index": int(i), "text": candidates[i], "score": float(scores[i])} for i in ranked]


@mcp.tool
def summarize_text(
    text: str, max_sentences: int = 3, max_tokens: int | None = None, diversity: float = 0.1
) -> str:
    """Produce an extractive (not generative) summary with selected source sentences."""
    _positive(max_sentences, "max_sentences", 100)
    _threshold(diversity, "diversity")
    if max_tokens is not None:
        _positive(max_tokens, "max_tokens")
    return tl_summarize_text(
        _one(text), max_sentences=max_sentences, max_tokens=max_tokens,
        diversity=diversity, embedder=_embedder(_EMBEDDING_MODEL),
    )


@mcp.tool
def summarize_paragraphs(
    paragraphs: list[str], max_sentences_per_paragraph: int = 2, max_tokens: int | None = None
) -> list[str]:
    """Extractively summarize each paragraph; optionally apply a shared approximate token limit."""
    _positive(max_sentences_per_paragraph, "max_sentences_per_paragraph", 100)
    if max_tokens is not None:
        _positive(max_tokens, "max_tokens")
    return tl_summarize_paragraphs(
        _texts(paragraphs, allow_empty=True),
        max_sentences_per_paragraph=max_sentences_per_paragraph,
        max_tokens=max_tokens,
        embedder=_embedder(_EMBEDDING_MODEL),
    )


@mcp.tool
def rerank_passages(query: str, passages: list[str], top_k: int = 5) -> list[str]:
    """Rerank passages with a cross-encoder; returns passages in relevance order (no raw scores)."""
    _positive(top_k, "top_k", _MAX_TEXTS)
    return tl_rerank_passages(
        _one(query, "query"), _texts(passages, allow_empty=True),
        top_k=top_k, scorer=_reranker(_RERANK_MODEL),
    )


@mcp.tool
def extract_keyphrases(
    texts: list[str],
    top_k: int = 20,
    min_occurrences: int = 1,
    min_ngram: int = 1,
    max_ngram: int = 4,
    diversity: float = 0.15,
) -> list[dict[str, Any]]:
    """Extract scored, frequency-aware keyphrases with configurable n-gram size."""
    _positive(top_k, "top_k", 200)
    _positive(min_occurrences, "min_occurrences", _MAX_TEXTS)
    _positive(min_ngram, "min_ngram", 10)
    _positive(max_ngram, "max_ngram", 10)
    if max_ngram < min_ngram:
        raise ValueError("max_ngram must be >= min_ngram")
    _threshold(diversity, "diversity")
    return [
        asdict(item)
        for item in tl_extract_keyphrases(
            _texts(texts, allow_empty=True), top_k=top_k,
            min_occurrences=min_occurrences, min_ngram=min_ngram,
            max_ngram=max_ngram, diversity=diversity,
            embedder=_embedder(_EMBEDDING_MODEL),
        )
    ]


@mcp.tool
def keyphrases_to_tags(keyphrases: list[dict[str, Any]], source: str = "textlite_keyphrases") -> list[dict]:
    """Convert output from extract_keyphrases into annotation-style tag dictionaries."""
    if len(keyphrases) > 200:
        raise ValueError("Too many keyphrases (max 200)")
    phrases = [Keyphrase(text=str(p["text"]), score=float(p["score"]), occurrences=int(p["occurrences"])) for p in keyphrases]
    return tl_keyphrases_to_tags(phrases, source=source[:100])


@mcp.tool
def extract_topics(
    texts: list[str],
    top_k_keyphrases: int = 80,
    max_topics: int = 20,
    min_topic_size: int = 2,
    similarity_threshold: float = 0.72,
) -> list[dict[str, Any]]:
    """Discover high-level topics by clustering extracted keyphrases."""
    _positive(top_k_keyphrases, "top_k_keyphrases", 200)
    _positive(max_topics, "max_topics", 100)
    _positive(min_topic_size, "min_topic_size", 100)
    _threshold(similarity_threshold, "similarity_threshold")
    return [
        asdict(topic)
        for topic in tl_extract_topics(
            _texts(texts, allow_empty=True), top_k_keyphrases=top_k_keyphrases,
            max_topics=max_topics, min_topic_size=min_topic_size,
            similarity_threshold=similarity_threshold,
            embedder=_embedder(_EMBEDDING_MODEL),
        )
    ]


@mcp.tool
def assign_entities_to_topics(
    topics: list[dict[str, Any]], entities: list[dict[str, Any]], max_entities_per_topic: int = 12
) -> list[dict[str, Any]]:
    """Attach entities by lightweight lexical matching to topics from extract_topics.

    topics entries need id, label, score, size, keyphrases.
    entities entries need name and optionally id, weight, importance_score.
    """
    _positive(max_entities_per_topic, "max_entities_per_topic", 100)
    if len(topics) > 100 or len(entities) > _MAX_TEXTS:
        raise ValueError("Too many topics or entities")
    topic_objects = [
        Topic(id=str(t["id"]), label=_one(str(t["label"]), "topic label"),
              score=float(t["score"]), size=int(t["size"]),
              keyphrases=_texts([str(k) for k in t["keyphrases"]], allow_empty=True))
        for t in topics
    ]
    for entity in entities:
        if "name" in entity:
            _one(str(entity["name"]), "entity name")
    return tl_assign_entities_to_topics(topic_objects, entities, max_entities_per_topic=max_entities_per_topic)


def main() -> None:
    parser = argparse.ArgumentParser(description="TextLite FastMCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default=os.getenv("MCP_TRANSPORT", "stdio"))
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8000")))
    args = parser.parse_args()
    if args.transport == "stdio":
        # Do not print to stdout in stdio mode: stdout belongs to the MCP protocol.
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
