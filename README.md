# TextLite MCP Server (FastMCP)

A runnable [FastMCP](https://gofastmcp.com/) server wrapping the public [TextLite](https://github.com/meru-research/textlite) Python API, plus two helpful embedding-based tools. Use it locally via **stdio** or deploy it via **Streamable HTTP** to publish in the [official MCP Registry](https://registry.modelcontextprotocol.io/).

**Status:** Working source distribution with test coverage and a Docker deployment template. You must install the dependencies and download the NLP models to run actual inference. The code has not been published to PyPI or the MCP Registry and no hosted endpoint is provided.

## Quick start (Python 3.11+)

Requires Python 3.11+, Git, an internet connection for installation and first model download, and sufficient memory to load PyTorch/Sentence Transformer models.

```bash
unzip textlite-mcp.zip
cd textlite-mcp
python3.11 -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/client_example.py          # connects in-process and calls tools
```

**The first real NLP tool call downloads a model from Hugging Face.** `server_info` loads no model. This project explicitly installs TextLite from `git+https://github.com/meru-research/textlite.git@main`, because the inspected upstream `pyproject.toml` has no declared runtime dependencies. Our `pyproject.toml` declares those runtime dependencies separately.

### Start as a local MCP server

```bash
textlite-mcp --transport stdio
# or
python -m textlite_mcp.server --transport stdio
```

The process is deliberately silent: the **stdio** stream is reserved for MCP messages. Add it to an MCP client (see below) instead of typing JSON at the terminal.

### Connect from Claude Desktop

Edit Claude Desktop's `claude_desktop_config.json` (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS; `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "textlite": {
      "command": "/ABSOLUTE/PATH/textlite-mcp/.venv/bin/python",
      "args": ["-m", "textlite_mcp.server", "--transport", "stdio"]
    }
  }
}
```

On Windows, substitute `C:\\path\\textlite-mcp\\.venv\\Scripts\\python.exe` for `command`. Merge the `textlite` entry into any existing `mcpServers` instead of replacing other servers. Restart your MCP client after changing configuration.

### Start as an HTTP server

```bash
textlite-mcp --transport http --host 127.0.0.1 --port 8000
# MCP endpoint: http://127.0.0.1:8000/mcp
```

In a second terminal:

```bash
python examples/client_example.py http://127.0.0.1:8000/mcp
```

An MCP endpoint is not a normal REST API: use an MCP client/Inspector, not a plain browser GET request, to test the protocol.

### Test a local server (scripts)

[`examples/textlite_client.py`](examples/textlite_client.py) wraps the tools in an async `TextLiteClient` (one method per tool, plus `compress_text` / `compress_file`, which split long text into paragraphs and compress it in batches when it exceeds the server's input limits). `server` can be `inprocess`, `stdio`, `http` (`$TEXTLITE_MCP_URL`, default `http://127.0.0.1:8000/mcp`) or a full MCP URL.

```bash
# Every tool once against a running HTTP server (exit code 1 on failure)
python examples/smoke_local_server.py --server http

# Compression examples on examples/test_compression.txt (or --file your.txt)
python examples/compression_example.py --server http --budgets 80 150 300 \
  --query "How does battery storage help solar power?"
```

```python
from textlite_client import TextLiteClient, format_compression_report

async with TextLiteClient("http") as tl:          # top-level await works in Jupyter
    result = await tl.compress_file("examples/test_compression.txt", num_tokens=150)
    print(format_compression_report(result))
```

Token counts in the reports use TextLite's approximate count (`\w+` words plus punctuation), not an LLM tokenizer.

## Available tools

| MCP tool | Main arguments | Result |
| --- | --- | --- |
| `server_info` | none | Model names, server version and input limits |
| `deduplicate_texts` | `texts`, `similarity_threshold=0.9` | `original_texts`, `unique_texts`, `groups`, `selected_indices` |
| `cluster_entities` | `entities`, `similarity_threshold=0.72`, `dedup_threshold=0.92` | List of clusters: label, members, representative, cohesion |
| `compress_contexts` | `paragraphs`, `num_tokens=250` | Representative contexts, approximately token-budgeted |
| `compress_query_contexts` | `query`, `paragraphs`, `num_tokens=250` | Query-relevant extracted contexts |
| `semantic_similarity` | `text_a`, `text_b` | Cosine similarity score between -1 and 1 |
| `semantic_search` | `query`, `texts`, `top_k=5` | Ranked matches with original cleaned indices and cosine scores |
| `summarize_text` | `text`, `max_sentences=3`, `max_tokens` | Extractive summary string |
| `summarize_paragraphs` | `paragraphs`, `max_sentences_per_paragraph=2`, `max_tokens` | Extractive summary per paragraph |
| `rerank_passages` | `query`, `passages`, `top_k=5` | Cross-encoder ranked passages (not scores) |
| `extract_keyphrases` | `texts`, `top_k=20`, n-gram controls | Keyphrases with score and occurrences |
| `keyphrases_to_tags` | `keyphrases`, `source` | Annotation-style tags |
| `extract_topics` | `texts`, `top_k_keyphrases=80`, `max_topics=20` | Topics with labels, keyphrases and scores |
| `assign_entities_to_topics` | `topics`, `entities`, `max_entities_per_topic=12` | Topics with lexically matched entities |

`semantic_similarity` and `semantic_search` are additional wrappers using TextLite's `SentenceTransformerEmbedder`; the other tools wrap TextLite's public functions. This is **extractive** summarization: it selects phrases/sentences from input rather than generating new prose.

## Sample use from Python (through MCP)

```python
import asyncio
from fastmcp import Client
from textlite_mcp.server import mcp

async def demo():
    async with Client(mcp) as client:
        result = await client.call_tool("semantic_search", {
            "query": "Neural search and embeddings",
            "texts": [
                "Sentence transformers encode meaning.",
                "A recipe for chocolate brownies.",
                "Vector databases store embeddings."
            ],
            "top_k": 2
        })
        print(result.data)

asyncio.run(demo())
```

## Docker deployment

```bash
docker compose up --build -d
python examples/client_example.py http://127.0.0.1:8000/mcp
```

Docker Compose exposes port 8000 **only on localhost** by default. The first inference can take longer because the model is downloaded and cached in the persistent `model_cache` volume. See [DEPLOYMENT.md](DEPLOYMENT.md) for TLS, authentication, reverse proxy, model caching, resource limits and production considerations.

## Publish to the official MCP Registry

The preferred route in this project is a **remote-only registry listing**. The registry stores metadata and a link to your already deployed HTTPS MCP server; it does **not** host your Python code. A remote-only listing needs no PyPI release. You need a publicly accessible HTTPS `/mcp` endpoint you control and ownership of the namespace you publish under.

```bash
# 1. Deploy the Docker container behind an HTTPS reverse proxy, e.g.:
#    https://mcp.example.com/mcp

# 2. Create registry metadata for a GitHub namespace you own:
python scripts/generate_registry.py \
  --github-user YOUR_GITHUB_USERNAME \
  --url https://mcp.example.com/mcp \
  --version 0.1.0

# 3. Install the official publisher, e.g. on macOS:
brew install mcp-publisher

# 4. Log in, validate and publish:
mcp-publisher login github
mcp-publisher validate server.json
mcp-publisher publish server.json
```

For non-macOS installations use official publisher binaries from [modelcontextprotocol/registry releases](https://github.com/modelcontextprotocol/registry/releases). Replace both placeholders; the sample metadata is **not publishable unchanged**. See [DEPLOYMENT.md](DEPLOYMENT.md) for an end-to-end procedure.

## Runtime configuration

Settings can be exported as environment variables; see `.env.example`:

| Variable | Default |
| --- | --- |
| `MCP_TRANSPORT` | `stdio` |
| `MCP_HOST` | `127.0.0.1` |
| `MCP_PORT` | `8000` |
| `TEXTLITE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `TEXTLITE_QUERY_MODEL` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| `TEXTLITE_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `TEXTLITE_MAX_TEXTS` | `100` |
| `TEXTLITE_MAX_CHARS_PER_TEXT` | `20000` |
| `TEXTLITE_MAX_TOTAL_CHARS` | `120000` |

The guards limit inputs but do **not** provide authentication, rate limiting or runtime isolation. See `DEPLOYMENT.md` before exposing the service publicly.

## Notes / caveats

- TextLite's `num_tokens` uses its default *cheap approximate token count*, not the tokenizer of a specific LLM. Some upstream code has a fallback that may select a first item over the budget. Check actual LLM token counts if strict budgets are required.
- Deduplication and clustering use dense similarity matrices (roughly quadratic in input count), so the input cap matters. Tune it conservatively.
- TextLite topic extraction creates topic clusters from English-style keyphrase candidates. Entity-to-topic association is lexical, not semantic.
- `rerank_passages` returns ranked text, not calibrated relevance probabilities.
- The semantic similarity scores are cosine scores, not probabilities. Embedding results depend on the model selected.
- Remote deployment needs HTTPS, auth, ingress rate limits, observability, and consideration of whether your submitted text is sensitive. By default inference is local to the server, but first-use model downloads require network access.
- The upstream TextLite dependency is installed from the mutable `main` branch. For reproducible production builds, pin to a reviewed commit SHA in `pyproject.toml` and redeploy.
- The project's Git dependency is allowed by `pip` for source installs but **cannot be published to PyPI as-is**. Either publish an appropriately licensed TextLite version to PyPI under an owned distribution name or change the packaging model before attempting PyPI publication. Remote-only MCP Registry publishing does not require PyPI.

## Project layout

```text
src/textlite_mcp/server.py         FastMCP server and all tools
src/textlite_mcp/__init__.py       Python package version
examples/client_example.py        In-memory / HTTP MCP client example
examples/claude_desktop_config.json  Example client config
scripts/generate_registry.py      Create server.json for MCP Registry
server.json.example               Documentation template
Dockerfile / docker-compose.yml   Deployment
.circleci/config.yml              Optional CI test/build workflow
tests/                             Automated tests without downloading models
DEPLOYMENT.md                      Hosting and registry publication guide
```

## Sources

- TextLite API: https://github.com/meru-research/textlite
- Standalone FastMCP: https://gofastmcp.com/docs/deployment/running-server
- Registry remote servers: https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/remote-servers.mdx
- Publisher CLI: https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/cli/commands.md
- CI packaging tutorial: https://circleci.com/blog/building-and-deploying-a-python-mcp-server-with-fastmcp/

TextLite is a separate upstream project with its own Apache-2.0 license. This archive contains an independent integration, not a vendored copy of TextLite.
