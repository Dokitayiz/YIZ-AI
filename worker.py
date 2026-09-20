"""Yiz AI worker process. Runs heavy media/document jobs outside the API process."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from typing import Any, Awaitable, Callable

import redis.asyncio as redis

import yiz_ai

log = logging.getLogger("yiz.worker")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

QUEUE_KEY = os.getenv("YIZ_JOB_QUEUE", "yiz:jobs")
JOB_TTL = int(os.getenv("YIZ_JOB_TTL", 3600))
REDIS_URL = os.getenv("REDIS_URL", "").strip()

HANDLERS: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "generate_qr": yiz_ai.generate_qr,
    "generate_image": yiz_ai._generate_image,
    "text_to_speech": yiz_ai._text_to_speech,
    "generate_video": yiz_ai._generate_video,
    "generate_text_file": yiz_ai._generate_text_file,
    "generate_csv": yiz_ai._generate_csv,
    "generate_json_file": yiz_ai._generate_json_file,
    "generate_markdown_file": yiz_ai._generate_markdown_file,
    "generate_pdf": yiz_ai._generate_pdf,
    "generate_xlsx": yiz_ai._generate_xlsx,
    "generate_docx": yiz_ai._generate_docx,
    "make_chart": yiz_ai._make_chart,
}


async def process(r: redis.Redis, raw: str) -> None:
    job = json.loads(raw)
    job_id = job["id"]
    name = job["name"]
    user_id = job["user_id"]
    handler = HANDLERS.get(name)
    key = f"yiz:job:{job_id}"
    if not handler:
        await r.hset(key, mapping={"status": "failed", "error": f"unknown job: {name}"})
        await r.expire(key, JOB_TTL)
        return

    token = yiz_ai.CURRENT_USER_ID.set(user_id)
    try:
        await r.hset(key, mapping={"status": "running"})
        result = await handler(**job.get("args", {}))
        if not isinstance(result, dict):
            result = {"result": result}
        await r.hset(key, mapping={"status": "succeeded", "result": json.dumps(result, default=str)})
        await r.expire(key, JOB_TTL)
    except Exception as exc:
        log.exception("job %s (%s) failed", job_id, name)
        await r.hset(key, mapping={"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        await r.expire(key, JOB_TTL)
    finally:
        yiz_ai.CURRENT_USER_ID.reset(token)


async def main() -> None:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required for the Yiz AI worker")
    await yiz_ai.db.connect()
    r = redis.from_url(REDIS_URL, decode_responses=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    log.info("Yiz AI worker listening on %s", QUEUE_KEY)
    try:
        while not stop.is_set():
            now = str(time.time())
            await r.set("yiz:worker:heartbeat", now, ex=30)
            item = await r.blpop(QUEUE_KEY, timeout=5)
            if not item:
                continue
            _, raw = item
            await process(r, raw)
    finally:
        await r.aclose()
        await yiz_ai.db.close()


if __name__ == "__main__":
    asyncio.run(main())
