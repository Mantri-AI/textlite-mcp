# Deployment and MCP Registry publication

## 1. Check your server locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
textlite-mcp --transport http --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
python examples/client_example.py http://127.0.0.1:8000/mcp
```

First inference downloads the configured models from Hugging Face and can use considerable RAM/disk. TextLite runs inference on your server, not in an external inference API. See `.env.example` for model identifiers.

## 2. Deploy the container

```bash
docker compose up --build -d
docker compose logs -f textlite-mcp
```

The default Compose port mapping is `127.0.0.1:8000:8000` to prevent accidental public exposure. For a VPS with a reverse proxy on the same host, keep this mapping.

For a production service, make sure:

- The host has enough RAM, CPU and disk for PyTorch, cached embedding and cross-encoder models. Starting with 2+ GB RAM for light CPU use is a rough lower bound; larger inputs and concurrent requests can need substantially more.
- Outbound access to GitHub (build) and Hugging Face (first inference) is available; or pre-fetch and bake models into the image for offline use.
- Only trusted clients reach your service. Use HTTPS, user/client authentication, per-client rate limiting, text-size limits, logging with sensitive text redaction, and request/concurrency timeouts.
- You monitor resource use and mitigate abuse from repeated expensive embedding requests. HTTP mode in this sample is **not authenticated**.
- You manage TextLite and model version changes by pinning the upstream Git commit and model revisions, not just by leaving `main` as a mutable branch.

### Example HTTPS reverse proxy with Caddy

On the server install Caddy, configure a DNS A/AAAA record for `mcp.example.com` to your server, and use this Caddyfile:

```caddyfile
mcp.example.com {
    # ADD authentication / access gateway BEFORE allowing public requests.
    reverse_proxy 127.0.0.1:8000
}
```

Caddy can provision TLS certificates. **This snippet alone does not authenticate requests**: add an authentication gateway (e.g. OAuth/OIDC-capable reverse proxy or a managed identity-aware access layer). If your users only need localhost, skip public hosting altogether.

Once configured and protected, check the public endpoint with an MCP client:

```bash
python examples/client_example.py https://mcp.example.com/mcp
```

If your gateway requires Authorization headers, configure credentials in your MCP client (the bundled example uses an unauthenticated client). Do not hardcode tokens or passwords into the repository or example registry file. Use the remote server `headers` schema when advertising required credentials to clients.

## 3. Prepare registry metadata

The official MCP Registry indexes server metadata: it doesn't deploy your container. A remote-only record needs an accessible Streamable HTTP endpoint; it need not refer to any PyPI package.

From the project directory:

```bash
python scripts/generate_registry.py \
    --github-user your-github-username \
    --url https://mcp.example.com/mcp \
    --version 0.1.0
cat server.json
```

The resulting file declares:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "io.github.your-github-username/textlite-mcp",
  "title": "TextLite MCP",
  "description": "TextLite semantic NLP tools: compression, clustering, deduplication, summarization, similarity, keyphrases and topics",
  "version": "0.1.0",
  "remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/mcp"}]
}
```

To publish in a GitHub namespace, sign in as the owner of that GitHub account. To publish under a domain namespace (`com.example/...`), use the publisher's supported DNS/HTTP ownership-verification flow instead.

### When your remote needs authentication

Add a `headers` entry to the remote metadata describing the required header (no secret value). For example, inside the `remotes[0]` object:

```json
"headers": [
  {"name": "Authorization", "description": "Bearer access token for your account", "isRequired": true, "isSecret": true}
]
```

Check that your selected client and identity provider support that setup. A static bearer-token header should only be used if your ingress actually validates it; avoid exposing one shared token for a public multi-user service.

## 4. Publish to registry

Install the **official** `mcp-publisher` (for example `brew install mcp-publisher` on macOS, or download from official [GitHub releases](https://github.com/modelcontextprotocol/registry/releases)). Then:

```bash
mcp-publisher login github
mcp-publisher validate server.json
mcp-publisher publish server.json
```

Verify success using the publisher response and the search/discovery UI/API at https://registry.modelcontextprotocol.io/. Publishing an entry requires that you own the namespace, and the public endpoint must be accessible to clients. The registry may still be in preview, so consult official docs if flags/schema change.

**Do not publish `server.json.example` unchanged.** The `YOUR_*` placeholders are not real accounts or hosts.

For a new release: update the application version, deploy it, then generate new `server.json` with an incremented version and republish. Registry versions are versioned metadata entries; do not assume overwriting an older version is supported.

## 5. Local package distribution vs. PyPI distribution

This archive installs TextLite directly from GitHub via the `pyproject.toml` dependency:

```toml
"textlite @ git+https://github.com/meru-research/textlite.git@main"
```

This is fine for local `pip` builds and Docker. It is **not** a valid public PyPI dependency format for a package uploaded to PyPI under standard policy. To distribute a PyPI MCP package later, first make the upstream TextLite dependency installable from a package registry (subject to its license and publication rights), change this project's metadata to use that registry package, release your package, and follow the Registry's PyPI ownership-verification instructions (`mcp-name: <server name>` in the package README). This is optional and entirely separate from **remote-only** Registry publishing.

## 6. Optional CircleCI

`.circleci/config.yml` runs Python tests and builds the wheel/source distribution whenever attached to a CircleCI project. The configuration does **not** automatically push images, publish to PyPI, deploy hosting, or create registry records. Configure those release actions with appropriate protected CI credentials only after validating the production environment.

## Security checklist before public deployment

- Require authentication and authorization at the HTTP ingress (sample has none).
- Apply TLS, quotas, request size ceilings and strict resource timeouts.
- Isolate the runtime and run without root privileges (Dockerfile uses a non-root account).
- Pin dependency commits and model revisions; review supply-chain licenses and vulnerability updates.
- Avoid logging raw text, if it may contain private data.
- Review TextLite's algorithmic complexity for the intended workload.
- Keep an operational rollback and monitoring plan.

## Official references

- https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/remote-servers.mdx
- https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx
- https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/cli/commands.md
- https://gofastmcp.com/docs/deployment/running-server
