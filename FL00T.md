# Floot deployment steps

Floot deployment steps — fill in from Floot's documentation.

<!-- Paste Floot-specific configuration below this line. -->

---

## Generic requirements Floot must satisfy

- Python 3.12 runtime, or Docker support (a `Dockerfile` is included).
- A Postgres connection string available to the process as `DATABASE_URL`.
- A persistent volume mounted at `/app/media`, or a way to set `MEDIA_DIR`
  to a writable path that survives redeploys.
- At least 512 MB RAM and 1 vCPU.
- The web process must listen on `$PORT` (defaults to 8000).
- Start command: `uvicorn yiz_ai:app --host 0.0.0.0 --port $PORT`
- Environment variables set from `.env.example`. `JWT_SECRET` and
  `DATABASE_URL` are mandatory; the app exits on boot without them.
- Outbound HTTPS to the configured LLM provider, and to Piston, Tavily,
  Stability, Replicate, and any MCP server that is enabled.
- Response streaming must not be buffered by the edge proxy. The app sends
  `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`; a proxy
  that buffers anyway will break SSE chat.
- Idle timeout of at least 120 seconds on the `/api/chat` route. The stream
  emits a `: ping` comment every 15 seconds to hold the connection open.

## Notes

-
-
-
