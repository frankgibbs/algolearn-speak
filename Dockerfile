# The speak daemon (docs/DESIGN_REMOTE_DAEMON.md section 7): Telegram calls, Kokoro TTS and
# faster-whisper STT on GPU0 of patrick. Built and started only by the owner:
#   docker --context patrick compose up --build -d
#
# CUDA 12.6 runtime: cuBLAS 12 for ctranslate2 (faster-whisper; it does not use cuDNN), and the
# same CUDA major as the torch 2.14.0+cu126 wheel locked in uv.lock.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: py-tgcalls plays the opener file through it. espeak-ng: misaki's out-of-vocabulary G2P.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached across code changes), exactly as locked.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra daemon --no-install-project --python 3.12

# The code. .dockerignore keeps .env, .git, .venv, tests and docs out of the image.
COPY speak_server.py speak_protocol.py speak_engines.py speak_telegram.py speak_telegram_daemon.py speak_tone.py README.md LICENSE ./
RUN uv sync --frozen --extra daemon --python 3.12

ENV PATH=/opt/venv/bin:$PATH
CMD ["speak-telegram"]
