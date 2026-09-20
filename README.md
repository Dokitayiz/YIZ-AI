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
