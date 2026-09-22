FROM node:24-bookworm-slim@sha256:0e0ff40c39bc087845bfb27465a0df4ea419520094bc35842ff83dd8cbe6f9b6 AS frontend
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc AS uv
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime
COPY --from=uv /uv /usr/local/bin/uv
RUN useradd --create-home --uid 10001 modelport && mkdir /data /app && chown modelport:modelport /data /app
WORKDIR /app
COPY --chown=modelport:modelport pyproject.toml uv.lock README.md hatch_build.py LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --extra cpu --no-dev --no-install-project --link-mode=copy
COPY --chown=modelport:modelport src/ ./src/
COPY --from=frontend --chown=modelport:modelport /web/dist /app/web/dist
RUN --mount=type=cache,target=/root/.cache/uv MODELPORT_REQUIRE_WEB=1 uv sync --locked --extra cpu --no-dev --no-editable --link-mode=copy
ENV PATH="/app/.venv/bin:$PATH" MODELPORT_DATA_DIR=/data MODELPORT_WEB_DIR=/app/web/dist PYTHONUNBUFFERED=1
USER 10001:10001
EXPOSE 8765
VOLUME /data
HEALTHCHECK --interval=15s --start-period=90s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health/ready', timeout=3)"
CMD ["modelport", "serve", "--host", "0.0.0.0"]

FROM runtime AS test
USER root
COPY --chown=modelport:modelport tests/ ./tests/
COPY --chown=modelport:modelport scripts/ ./scripts/
COPY --chown=modelport:modelport examples/ ./examples/
RUN uv sync --locked --extra cpu --no-editable --no-cache
USER 10001:10001
HEALTHCHECK NONE
CMD ["python", "-m", "pytest", "-q", "--junitxml=/data/pytest-linux.xml"]

# Keep the deployable runtime as the default build target.
FROM runtime AS final
