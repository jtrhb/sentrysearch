FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies (cached layer)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || \
    uv sync --no-dev --no-install-project

# Copy source and install project
COPY . .
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "sentrysearch.api:app", "--host", "0.0.0.0", "--port", "8080"]
