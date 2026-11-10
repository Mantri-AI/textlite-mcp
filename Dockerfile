FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Git is required for installing the TextLite upstream repository.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --upgrade pip && python -m pip install . \
    && mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache
USER app
EXPOSE 8000
CMD ["textlite-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
