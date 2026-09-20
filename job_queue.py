"""Redis-backed job queue used to move heavy Yiz AI work out of the API process."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Optional

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "").strip()
QUEUE_KEY = os.getenv("YIZ_JOB_QUEUE", "yiz:jobs")
JOB_TTL = int(os.getenv("YIZ_JOB_TTL", 3600))
WAIT_TIMEOUT = int(os.getenv("YIZ_JOB_WAIT_TIMEOUT", 900))


def _client() -> redis.Redis:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required for worker jobs")
    return redis.from_url(REDIS_URL, decode_responses=True)


async def enqueue(name: str, args: dict[str, Any], user_id: str) -> str:
    r = _client()
    job_id = uuid.uuid4().hex
    payload = {"id": job_id, "name": name, "args": args, "user_id": user_id,
               "created_at": time.time()}
    try:
        await r.hset(f"yiz:job:{job_id}", mapping={
            "status": "queued", "name": name, "user_id": user_id,
            "created_at": str(payload["created_at"])
        })
        await r.expire(f"yiz:job:{job_id}", JOB_TTL)
        await r.rpush(QUEUE_KEY, json.dumps(payload, default=str))
        return job_id
    finally:
        await r.aclose()


async def get_job(job_id: str) -> Optional[dict[str, Any]]:
    r = _client()
    try:
        data = await r.hgetall(f"yiz:job:{job_id}")
        if not data:
            return None
        if data.get("result"):
            try:
                data["result"] = json.loads(data["result"])
            except json.JSONDecodeError:
                pass
        return data
    finally:
        await r.aclose()


async def worker_available(max_age: int = 30) -> bool:
    """Return whether a worker heartbeat is fresh enough to accept jobs."""
    r = _client()
    try:
        value = await r.get("yiz:worker:heartbeat")
        if not value:
            return False
        return (time.time() - float(value)) <= max_age
    except Exception:
        return False
    finally:
        await r.aclose()


async def enqueue_and_wait(name: str, args: dict[str, Any], user_id: str,
                           timeout: int = WAIT_TIMEOUT) -> dict[str, Any]:
    job_id = await enqueue(name, args, user_id)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = await get_job(job_id)
        if job is None:
            return {"error": "worker job disappeared"}
        status = job.get("status")
        if status == "succeeded":
            return job.get("result") or {}
        if status == "failed":
            return {"error": job.get("error", "worker job failed")}
        await asyncio.sleep(0.75)
    return {"error": "worker job timed out", "job_id": job_id}
