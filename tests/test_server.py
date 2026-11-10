"""Exercise TextLite's real operations with a lightweight deterministic fake embedder.

No model downloads, network calls, or GPU are needed for these tests.
"""

from __future__ import annotations

import re

from fastmcp import Client
import numpy as np
import pytest

import textlite_mcp.server as server


class DummyEmbedder:
    """Simple normalized bag-of-word hash encoder for fast tests."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            v = np.zeros(64)
            for word in re.findall(r"[a-z]+", text.lower()):
                idx = sum((i + 1) * ord(ch) for i, ch in enumerate(word)) % 64
                v[idx] += 1
            n = np.linalg.norm(v)
            vectors.append(v / n if n > 0 else v)
        return np.stack(vectors)


class DummyReranker:
    def score(self, query, passages):
        words = set(query.lower().split())
        return [sum(word in passage.lower().split() for word in words) for passage in passages]


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    monkeypatch.setattr(server, "_embedder", lambda _: DummyEmbedder())
    monkeypatch.setattr(server, "_reranker", lambda _: DummyReranker())


@pytest.mark.asyncio
async def test_tool_discovery_and_server_info():
    async with Client(server.mcp) as client:
        names = {tool.name for tool in await client.list_tools()}
        assert {"server_info", "compress_contexts", "deduplicate_texts", "cluster_entities", "semantic_search", "extract_topics"} <= names
        info = (await client.call_tool("server_info", {})).data
        assert info["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_dedup_and_cluster():
    async with Client(server.mcp) as client:
        result = (await client.call_tool("deduplicate_texts", {
            "texts": ["Neural search", "Neural search", "Banana pancakes"]
        })).data
        assert len(result["unique_texts"]) == 2
        assert any(len(group) == 2 for group in result["groups"])
        clusters = (await client.call_tool("cluster_entities", {"entities": ["NLP", "NLP"]})).data
        assert len(clusters) == 1
        assert "representative" in clusters[0]


@pytest.mark.asyncio
async def test_similarity_and_search():
    async with Client(server.mcp) as client:
        same = (await client.call_tool("semantic_similarity", {"text_a": "apple tree", "text_b": "apple tree"})).data
        assert same["score"] == pytest.approx(1.0)
        matches = (await client.call_tool("semantic_search", {
            "query": "apple tree", "texts": ["pear jam", "apple tree", "banana"], "top_k": 2
        })).data
        assert matches[0]["text"] == "apple tree"
        assert len(matches) == 2


@pytest.mark.asyncio
async def test_compression_and_summaries():
    async with Client(server.mcp) as client:
        result = (await client.call_tool("compress_contexts", {"paragraphs": ["Apple trees grow fruit."], "num_tokens": 50})).data
        assert result
        query_result = (await client.call_tool("compress_query_contexts", {
            "query": "apple trees", "paragraphs": ["Apple trees grow fruit."], "num_tokens": 50, "min_similarity": 0
        })).data
        assert query_result
        summary = (await client.call_tool("summarize_text", {
            "text": "Apples grow on trees. Bananas grow on plants. Trees also give shade.", "max_sentences": 2
        })).data
        assert isinstance(summary, str) and summary
        paragraph_summaries = (await client.call_tool("summarize_paragraphs", {
            "paragraphs": ["Apples grow on trees. Trees are plants."], "max_sentences_per_paragraph": 1
        })).data
        assert paragraph_summaries


@pytest.mark.asyncio
async def test_keyphrases_topics_and_assignments():
    async with Client(server.mcp) as client:
        phrases = (await client.call_tool("extract_keyphrases", {
            "texts": ["Neural search improves semantic retrieval for documents."], "top_k": 5,
        })).data
        assert phrases and "occurrences" in phrases[0]
        tags = (await client.call_tool("keyphrases_to_tags", {"keyphrases": phrases})).data
        assert tags[0]["type"] == "KEYPHRASE"
        topics = (await client.call_tool("extract_topics", {
            "texts": ["Neural search enables semantic retrieval and ranking in vector databases."],
            "top_k_keyphrases": 8, "min_topic_size": 1
        })).data
        assert isinstance(topics, list)
        assigned = (await client.call_tool("assign_entities_to_topics", {
            "topics": [{"id": "topic_1", "label": "neural search", "score": 1.0, "size": 1, "keyphrases": ["neural search"]}],
            "entities": [{"id": "e1", "name": "neural search", "weight": 2}]
        })).data
        assert assigned[0]["entities"][0]["name"] == "neural search"


@pytest.mark.asyncio
async def test_reranking():
    async with Client(server.mcp) as client:
        ranked = (await client.call_tool("rerank_passages", {
            "query": "vector search", "passages": ["banana bread", "vector search technique"], "top_k": 1
        })).data
        assert ranked == ["vector search technique"]


@pytest.mark.asyncio
async def test_validation_is_returned_as_tool_error():
    async with Client(server.mcp) as client:
        result = await client.call_tool("deduplicate_texts", {
            "texts": ["a", "b"], "similarity_threshold": 1.5
        }, raise_on_error=False)
        assert result.is_error
