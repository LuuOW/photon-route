FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PHOTON_EVAL_CACHE=/app/.cache/eval

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /app/
COPY src   /app/src
COPY eval  /app/eval
COPY space /app/space

RUN pip install --upgrade pip \
 && pip install -e ".[photonic,training]"

# Train v2: fetches arxiv abstracts (per rule, never stored in repo;
# fetched at build time and SHA-verified against the frozen manifest),
# runs the InfoNCE + Bhattacharyya trainer, dumps weights.npz consumable
# by the v2 numpy encoder. Falls back to SHA-init at serve time if the
# train step fails so the container always boots.
RUN python -m space.train --out /app/weights.npz --steps 100 \
 || (echo "[build] training failed; container will serve sha_init only" && rm -f /app/weights.npz)

EXPOSE 7860

CMD ["uvicorn", "photon_route.http_server:app", "--host", "0.0.0.0", "--port", "7860"]
