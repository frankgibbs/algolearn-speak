# The speak daemon (docs/DESIGN_REMOTE_DAEMON.md section 7): Telegram calls, Kokoro TTS and
# faster-whisper STT on GPU0 of patrick. Built and started only by the owner, on patrick, from
# the deploy directory (an rsync of this repo, with .env next to docker-compose.yml):
#   ssh frankg@192.168.86.31 'cd ~/algolearn-speak && docker compose up --build -d'
#
# CUDA 12.6 runtime base, the same CUDA major as the torch 2.14.0+cu126 wheel locked in uv.lock.
# ctranslate2 (faster-whisper) needs cuBLAS 12 and cuDNN 9; this base has no cuDNN, so both come
# from the nvidia-*-cu12 pip packages that torch installs, put on LD_LIBRARY_PATH below so the
# loader finds them regardless of import order.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: py-tgcalls plays the opener file through it. espeak-ng: misaki's out-of-vocabulary G2P
# (the daemon refuses to start without it). tzdata: TZ=America/Los_Angeles must resolve; without
# zoneinfo glibc silently runs on UTC and quiet hours would be off by hours.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached across code changes), exactly as locked: --locked fails the build
# when uv.lock does not match pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra daemon --no-install-project --python 3.12

# The code. .dockerignore keeps .env, .git, .venv, tests and docs out of the build context.
COPY speak_server.py speak_protocol.py speak_engines.py speak_personas.py speak_telegram.py speak_telegram_daemon.py speak_tone.py README.md LICENSE ./
RUN uv sync --locked --extra daemon --no-editable --python 3.12

ENV PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
CMD ["speak-telegram"]
