# Yiz AI

Single-file FastAPI assistant: streaming chat with tool calling, JWT auth with
rotating refresh tokens, media and document generation, and DuckDB analysis.

## Run locally

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env`, then set `DATABASE_URL`, `JWT_SECRET`, and one LLM
   provider: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OLLAMA_API_KEY` /
   `OLLAMA_BASE_URL`. With more than one set, `LLM_PROVIDER` picks which is
   used; unset, priority is anthropic, then openai, then ollama.
4. `uvicorn yiz_ai:app --reload --port 8000`
5. Open http://localhost:8000 and create an account. Migrations run on boot.

## Deploy to any Docker PaaS

1. Push this folder to a repo and point the host at it; the `Dockerfile` is the build.
2. Provision Postgres and expose its DSN as `DATABASE_URL`.
3. Set every variable from `.env.example` in the host's dashboard.
   `JWT_SECRET` is mandatory when `ENV=production` — the app refuses to boot without it.
4. Mount a persistent volume at `/app/media`, or point `MEDIA_DIR` at writable storage.
   Without it, generated files and uploads vanish on each redeploy.
5. Expose port 8000, or let the host inject `$PORT` and use the `Procfile` instead.
6. Set `ALLOWED_ORIGINS` to your real origin, and `CROSS_ORIGIN=true` only when the
   frontend is served from a different origin than the API.
7. Verify `GET /api/health` returns `{"ok": true}` with your provider and model.

Minimum: 512 MB RAM, 1 vCPU. Matplotlib and DuckDB are the memory-hungry paths.
Floot-specific steps are in `FL00T.md`.


## Production hardening in v4.1

- Docker now listens on the platform-provided `PORT` instead of hard-coding 8000.
- Container health checks follow the same runtime port.
- Added `/api/ready` for dependency-aware readiness checks.
- Added transient Postgres connection retries and serialized migrations.
- Refresh-token consumption is atomic to reduce concurrent-reuse races.
- Chat payloads and tool context are bounded to reduce accidental resource exhaustion.
- MCP/tool failures are contained so one failing integration does not tear down the SSE stream.
- Safer default CORS behavior: set `ALLOWED_ORIGINS` explicitly when using a separate frontend.
- Added security headers and deployment documentation.
- Added a Render Blueprint configuration.

## Yiz AI 5.0 architecture

The project now supports a split production topology:

- `frontend/` — standalone Nginx UI with `/api/*` reverse proxy and SSE-safe settings.
- `yiz_ai.py` — FastAPI API/orchestrator.
- `worker.py` — Redis-backed background worker for heavy media/document jobs.
- `job_queue.py` — queue/status client.
- S3-compatible object storage — uploads and generated media.
- Postgres — application state and media ownership metadata.
- Redis/Render Key Value — rate limiting and worker queue.

For local development, use `docker compose up --build` and open `http://localhost:8080`.

For production, configure `MEDIA_STORAGE=s3` and the S3/R2 credentials. The included `render.yaml` provisions the frontend, private API, worker, Redis-compatible Key Value, and Postgres resources.
