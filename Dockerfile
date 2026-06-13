# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into an isolated prefix so they can be copied cleanly
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY guardrails.py \
     policy_corpus.py \
     observability.py \
     output.py \
     agent_loop.py \
     api.py \
     telemetry.py \
     otel_exporter.py \
     console.py \
     logger.py \
     ./

# SQLite corpus DB lives on a mounted volume so data persists across restarts.
# The volume is mounted at /data; point DB_PATH there via env var.
ENV DB_PATH=/data/corpus.db
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create the data directory (overridden by volume mount at runtime)
RUN mkdir /data

EXPOSE 8000

# Seed the DB on startup, then launch the API
CMD ["sh", "-c", "python policy_corpus.py && uvicorn api:app --host 0.0.0.0 --port 8000"]
