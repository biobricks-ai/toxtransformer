# docker build -t biobricks-ai/cvae .
# docker run -p 6515:6515 -v .:/chemsim --rm --gpus all -it --name chemsim biobricks-ai/cvae
# docker run -p 6515:6515 --rm --gpus all -it --name chemsim 010438487580.dkr.ecr.us-east-1.amazonaws.com/biobricks/chemprop-transformer
# curl "http://localhost:6515/predict?property_token=5042&inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
# Stage 1: builder
FROM python:3.11-slim-bookworm AS builder

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src

# Copy lock and pyproject for dependency install caching
COPY pyproject.toml uv.lock ./

# Install dependencies in isolated env
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy app sources
COPY flask_cvae flask_cvae
COPY brick brick
COPY cvae cvae
COPY flask_cvae/requirements.txt requirements.txt

# Sync again to include any extras (like from requirements)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Stage 2: runtime
FROM nvidia/cuda:12.3.1-base-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    APP_DIR=/app \
    FLASK_APP=flask_cvae.app \
    ROOT_URL=http://localhost:6515 \
    PORT=6515 \
    PATH="/app/.venv/bin:$PATH"

# System deps
RUN apt-get update && apt-get install -y \
    git curl libxrender1 openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Copy uv runtime too
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy venv from builder
COPY --from=builder /src/.venv ${APP_DIR}/.venv

WORKDIR ${APP_DIR}

# Copy app code (plus DB/tokenizer files)
COPY flask_cvae flask_cvae
COPY brick/moe brick/moe
COPY brick/cvae.sqlite brick/cvae.sqlite
COPY brick/selfies_property_val_tokenizer brick/selfies_property_val_tokenizer
COPY cvae cvae

EXPOSE ${PORT}

# Launch using uv to ensure proper env
CMD ["uv", "run", "gunicorn", "-b", "0.0.0.0:${PORT}", "--timeout", "480", "--graceful-timeout", "480", "--workers", "1", "--keep-alive", "300", "flask_cvae.app:app"]
