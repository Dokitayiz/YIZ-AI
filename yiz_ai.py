#!/usr/bin/env python3
"""Yiz AI — single-file backend with chat, media gen, data analysis, n8n, MCP."""
from __future__ import annotations

import ast
import asyncio
import base64
import csv as _csv
import contextvars
import hashlib
import importlib.util
import inspect
import io
import ipaddress
import json
import logging
import mimetypes
import operator as _op
import os
import re
import secrets
import shutil
import socket
import tempfile
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Optional
from urllib.parse import quote, urlencode, urlparse
from xml.sax.saxutils import escape as _xml_escape

import asyncpg
import httpx
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from dotenv import load_dotenv
from fastapi import (Cookie, Depends, FastAPI, File as FastFile, Header,
                     HTTPException, Query, Request, Response, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, RedirectResponse,
                               StreamingResponse)
from pydantic import BaseModel, EmailStr, Field

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("yiz")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------- CONFIG ----------------
ENV = os.getenv("ENV", "development").lower()
IS_PROD = ENV == "production"

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "").strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required.")

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
if IS_PROD and not JWT_SECRET:
    raise RuntimeError("JWT_SECRET required in production.")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_urlsafe(48)
    log.warning("JWT_SECRET unset; generated an ephemeral development secret.")
JWT_ALG = "HS256"

ACCESS_TTL_MIN = int(os.getenv("ACCESS_TTL_MIN", 15))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", 30))
COOKIE_ACCESS = "yiz_access"
COOKIE_REFRESH = "yiz_refresh"

CROSS_ORIGIN = os.getenv("CROSS_ORIGIN", "false").lower() in ("1", "true", "yes")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
FRONTEND_URL = os.getenv("FRONTEND_URL", "").rstrip("/")
# Public origin of this service (Render sets RENDER_EXTERNAL_URL automatically).
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL")
                   or os.getenv("RENDER_EXTERNAL_URL", "")).strip().rstrip("/")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
if not OLLAMA_BASE_URL and OLLAMA_API_KEY:
    OLLAMA_BASE_URL = "https://ollama.com/v1"
# Kimi (Moonshot AI) and DeepSeek both speak the OpenAI chat-completions format,
# so they reuse _stream_openai() with their own base URL and key.
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "").strip()
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2-0711-preview")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# anthropic | openai | kimi | deepseek | ollama | "" (auto: tries each configured
# provider in _PROVIDER_ORDER until one has credentials)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
# Allow callers to pick any model the configured providers expose.
ALLOW_MODEL_OVERRIDE = os.getenv("ALLOW_MODEL_OVERRIDE", "false").lower() in ("1", "true", "yes")
# When an override is used, require the model to appear in that provider's live
# /api/models listing. Fails open (allows the request) if the listing call itself
# fails, so a flaky provider API can't block chat entirely.
MODEL_OVERRIDE_STRICT = os.getenv("MODEL_OVERRIDE_STRICT", "true").lower() in ("1", "true", "yes")
MODEL_LIST_TTL = int(os.getenv("MODEL_LIST_TTL", 300))
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# Audio transcription (voice notes) speaks OpenAI's /audio/transcriptions API;
# reuses OPENAI_API_KEY/OPENAI_BASE_URL. No effect without an OpenAI key.
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", 4096))

# Connection details for every OpenAI-compatible provider (all but anthropic).
# _stream_openai() and _list_provider_models() both key off this so adding a
# new OpenAI-compatible provider only ever needs an entry here plus _PROVIDERS.
_OPENAI_COMPAT: dict[str, tuple[str, str]] = {
    "openai": (OPENAI_BASE_URL, OPENAI_API_KEY),
    "kimi": (KIMI_BASE_URL, KIMI_API_KEY),
    "deepseek": (DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY),
    "ollama": (OLLAMA_BASE_URL or "https://ollama.com/v1", OLLAMA_API_KEY),
}

SANDBOX_PROVIDER = os.getenv("SANDBOX_PROVIDER", "piston").lower()
PISTON_URL = os.getenv("PISTON_URL", "https://emkc.org/api/v2/piston").rstrip("/")
E2B_API_KEY = os.getenv("E2B_API_KEY", "")
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", 30))
SANDBOX_RUNS_PER_MIN = int(os.getenv("SANDBOX_RUNS_PER_MIN", 10))

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").rstrip("/")
N8N_API_KEY = os.getenv("N8N_API_KEY", "")
MCP_SERVERS = os.getenv("MCP_SERVERS", "")
MCP_ALLOW_PRIVATE = os.getenv("MCP_ALLOW_PRIVATE", "false").lower() in ("1", "true", "yes")
MCP_MAX_RESPONSE_BYTES = int(os.getenv("MCP_MAX_RESPONSE_BYTES", 2_000_000))
MCP_TIMEOUT = float(os.getenv("MCP_TIMEOUT", 30))
PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR", "plugins"))
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))
MEDIA_CACHE_DIR = Path(os.getenv("MEDIA_CACHE_DIR", str(MEDIA_DIR / "cache")))
MEDIA_STORAGE = os.getenv("MEDIA_STORAGE", "local").strip().lower()
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip()
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_REGION = os.getenv("S3_REGION", "auto").strip()
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "yiz-ai").strip("/")
MEDIA_MAX_BYTES = int(os.getenv("MEDIA_MAX_BYTES", 25 * 1024 * 1024))

WEB_FETCH_MAX_REDIRECTS = 5
WEB_FETCH_MAX_BYTES = 5 * 1024 * 1024
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", 60))
DUCKDB_ROW_LIMIT = 200
DB_CONNECT_RETRIES = max(1, int(os.getenv("DB_CONNECT_RETRIES", 8)))
DB_CONNECT_TIMEOUT_SEC = max(2, int(os.getenv("DB_CONNECT_TIMEOUT_SEC", 10)))


def _ensure_dir(path: Path, label: str, required: bool = False) -> None:
    """Create a working directory, optionally failing loudly."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if required:
            raise RuntimeError(f"{label} {path} is not writable: {exc}") from exc
        log.warning("%s %s is not writable: %s", label, path, exc)


if MEDIA_STORAGE not in {"local", "s3"}:
    raise RuntimeError("MEDIA_STORAGE must be 'local' or 's3'.")
if MEDIA_STORAGE == "s3" and not S3_BUCKET:
    raise RuntimeError("S3_BUCKET is required when MEDIA_STORAGE=s3.")
_ensure_dir(MEDIA_DIR, "MEDIA_DIR", required=MEDIA_STORAGE == "local")
_ensure_dir(MEDIA_CACHE_DIR, "MEDIA_CACHE_DIR", required=False)
_ensure_dir(PLUGINS_DIR, "PLUGINS_DIR")

# Request-local identity used by media storage and background jobs.
CURRENT_USER_ID: contextvars.ContextVar[str] = contextvars.ContextVar("yiz_user_id", default="system")

RL_LOGIN_PER_MIN = int(os.getenv("RL_LOGIN_PER_MIN", 8))
RL_SIGNUP_PER_MIN = int(os.getenv("RL_SIGNUP_PER_MIN", 4))
RL_API_PER_MIN = int(os.getenv("RL_API_PER_MIN", 120))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", "").rstrip("/")

ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "true").lower() in ("1", "true", "yes")
MAX_STEPS = int(os.getenv("MAX_STEPS", 6))
PORT = int(os.getenv("PORT", 8000))

SYSTEM_PROMPT = """You are Yiz AI, a capable assistant for chat, code, files, and data.

Tools — use them when they help:
- `calculator` for arithmetic.
- `run_code` to execute code in a sandbox (python, javascript, typescript, go, rust, java, c, cpp, bash, sql, ruby, php).
- `web_search` / `web_fetch` for current info or a specific URL.
- `generate_image` to create images. `text_to_speech` for audio. `generate_video` for short clips.
- File generators: `generate_text_file` (any extension), `generate_csv`, `generate_json_file`,
  `generate_markdown_file`, `generate_pdf`, `generate_xlsx`, `generate_docx`. Use these whenever
  the user asks for a downloadable file, report, dataset, spreadsheet, or document.
- Data analysis: `analyze_data` (profile a CSV/Parquet/JSON), `query_data` (SQL via DuckDB),
  `make_chart` (line/bar/scatter/pie PNG). Use these for any data task.
- `n8n_trigger` for automation. `mcp_call` for MCP servers. `save_note` for persistent facts.

Rules:
- Never claim an action you didn't perform via a tool.
- For data questions, prefer `analyze_data` then `query_data` then `make_chart` — show numbers, then picture.
- For document requests, prefer the file-generator tools and return the download URL.
- When you generate media, describe what you're creating in one line.
- Be concise. Direct answers over preamble."""

# ---------------- MIGRATIONS ----------------
MIGRATIONS = [
    (1, "init", """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            name TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS messages (
            id BIGSERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tool_calls JSONB,
            tool_call_id TEXT,
            name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
        CREATE TABLE IF NOT EXISTS notes (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
        CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id, id DESC);
        CREATE TABLE IF NOT EXISTS code_runs (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            language TEXT NOT NULL,
            code TEXT NOT NULL,
            stdout TEXT,
            stderr TEXT,
            exit_code INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
        CREATE INDEX IF NOT EXISTS idx_code_runs_user ON code_runs(user_id, id DESC);
    """),
    (2, "oauth", """
        ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_provider TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_sub TEXT;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oauth ON users(oauth_provider, oauth_sub)
            WHERE oauth_provider IS NOT NULL AND oauth_sub IS NOT NULL;
    """),
    (3, "sessions_refresh", """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL,
            revoked BOOLEAN NOT NULL DEFAULT FALSE,
            user_agent TEXT,
            ip TEXT);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            token_hash TEXT UNIQUE NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL,
            used_at TIMESTAMPTZ,
            replaced_by TEXT);
        CREATE INDEX IF NOT EXISTS idx_refresh_session ON refresh_tokens(session_id);
    """),
    (4, "attachments", """
        CREATE TABLE IF NOT EXISTS attachments (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            content_type TEXT,
            size_bytes BIGINT,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
        CREATE INDEX IF NOT EXISTS idx_attachments_user
            ON attachments(user_id, created_at DESC);
    """),
    (5, "media_metadata", """
        CREATE TABLE IF NOT EXISTS media_objects (
            stored_name TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            content_type TEXT,
            size_bytes BIGINT,
            original_name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_accessed_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS idx_media_objects_user
            ON media_objects(user_id, created_at DESC);
    """),
]


async def run_migrations(conn: asyncpg.Connection) -> None:
    """Apply unapplied migrations while serializing startup across replicas."""
    await conn.execute("SELECT pg_advisory_lock(874321009)")
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _schema_version (
                version INT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
        current = await conn.fetchval("SELECT COALESCE(MAX(version),0) FROM _schema_version")
        for version, name, sql in MIGRATIONS:
            if version <= current:
                continue
            log.info("applying migration v%03d %s", version, name)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("INSERT INTO _schema_version (version, name) VALUES ($1,$2)",
                                   version, name)
    finally:
        await conn.execute("SELECT pg_advisory_unlock(874321009)")


# ---------------- DATABASE ----------------
def _norm_dsn(dsn: str) -> str:
    """Normalize a postgres:// DSN to the postgresql:// form asyncpg expects."""
    if dsn.startswith("postgres://"):
        return "postgresql://" + dsn[len("postgres://"):]
    return dsn


class Database:
    """Thin asyncpg data-access layer for every table the app owns."""
    def __init__(self, dsn: str) -> None:
        """Store the normalized DSN."""
        self.dsn = _norm_dsn(dsn)
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Open the pool and bring the schema up to date, retrying transient DB startup failures."""
        last_error: Optional[Exception] = None
        for attempt in range(DB_CONNECT_RETRIES):
            try:
                self.pool = await asyncpg.create_pool(
                    self.dsn, min_size=1, max_size=10,
                    command_timeout=30, timeout=DB_CONNECT_TIMEOUT_SEC)
                async with self.pool.acquire() as conn:
                    await run_migrations(conn)
                log.info("database connected")
                return
            except Exception as exc:
                last_error = exc
                if self.pool:
                    await self.pool.close()
                    self.pool = None
                delay = min(2 ** attempt, 10)
                log.warning("database connection attempt %d/%d failed: %s; retrying in %ss",
                            attempt + 1, DB_CONNECT_RETRIES, exc, delay)
                await asyncio.sleep(delay)
        raise RuntimeError(f"database unavailable after {DB_CONNECT_RETRIES} attempts: {last_error}")

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()

    async def create_user(self, email: str, password_hash: Optional[str],
                          name: Optional[str] = None,
                          oauth_provider: Optional[str] = None,
                          oauth_sub: Optional[str] = None) -> dict:
        """Insert a user and return its public fields."""
        uid = uuid.uuid4().hex
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                """INSERT INTO users (id,email,password_hash,name,oauth_provider,oauth_sub)
                   VALUES ($1,$2,$3,$4,$5,$6)
                   RETURNING id,email,name,is_active,created_at,oauth_provider""",
                uid, email.lower(), password_hash, name, oauth_provider, oauth_sub)
        return dict(row)

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        """Look up a user by email."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM users WHERE email=$1", email.lower())
        return dict(row) if row else None

    async def get_user(self, uid: str) -> Optional[dict]:
        """Look up a user by id."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM users WHERE id=$1", uid)
        return dict(row) if row else None

    async def get_user_by_oauth(self, provider: str, sub: str) -> Optional[dict]:
        """Look up a user by OAuth identity."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                "SELECT * FROM users WHERE oauth_provider=$1 AND oauth_sub=$2",
                provider, sub)
        return dict(row) if row else None

    async def link_oauth(self, uid: str, provider: str, sub: str) -> None:
        """Attach an OAuth identity to an existing local account."""
        async with self.pool.acquire() as c:
            await c.execute(
                "UPDATE users SET oauth_provider=$1, oauth_sub=$2 "
                "WHERE id=$3 AND oauth_provider IS NULL", provider, sub, uid)

    async def create_conversation(self, uid: str, title: str = "New chat") -> str:
        """Create a conversation and return its id."""
        cid = uuid.uuid4().hex
        async with self.pool.acquire() as c:
            await c.execute(
                "INSERT INTO conversations (id,user_id,title) VALUES ($1,$2,$3)",
                cid, uid, title[:80])
        return cid

    async def list_conversations(self, uid: str, limit: int = 100) -> list[dict]:
        """List a user's conversations, most recent first."""
        async with self.pool.acquire() as c:
            rows = await c.fetch(
                """SELECT id,title,created_at,updated_at FROM conversations
                   WHERE user_id=$1 ORDER BY updated_at DESC LIMIT $2""", uid, limit)
        return [dict(r) for r in rows]

    async def get_conversation(self, cid: str, uid: str) -> Optional[dict]:
        """Fetch a conversation owned by a user."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                "SELECT * FROM conversations WHERE id=$1 AND user_id=$2", cid, uid)
        return dict(row) if row else None

    async def delete_conversation(self, cid: str, uid: str) -> bool:
        """Delete a conversation owned by a user."""
        async with self.pool.acquire() as c:
            res = await c.execute(
                "DELETE FROM conversations WHERE id=$1 AND user_id=$2", cid, uid)
        return res.endswith("1")

    async def rename_conversation(self, cid: str, uid: str, title: str) -> None:
        """Retitle a conversation owned by a user."""
        async with self.pool.acquire() as c:
            await c.execute(
                """UPDATE conversations SET title=$1,updated_at=NOW()
                   WHERE id=$2 AND user_id=$3""", title[:80], cid, uid)

    async def add_message(self, cid: str, role: str, content: str = "",
                          tool_calls: Optional[list] = None,
                          tool_call_id: Optional[str] = None,
                          name: Optional[str] = None) -> None:
        """Append a message and touch the conversation."""
        async with self.pool.acquire() as c:
            await c.execute(
                """INSERT INTO messages (conversation_id,role,content,tool_calls,
                   tool_call_id,name) VALUES ($1,$2,$3,$4,$5,$6)""",
                cid, role, content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id, name)
            await c.execute("UPDATE conversations SET updated_at=NOW() WHERE id=$1", cid)

    async def get_messages(self, cid: str) -> list[dict]:
        """Load a conversation's messages in order."""
        async with self.pool.acquire() as c:
            rows = await c.fetch(
                """SELECT role,content,tool_calls,tool_call_id,name
                   FROM messages WHERE conversation_id=$1 ORDER BY id ASC""", cid)
        out = []
        for r in rows:
            m = {"role": r["role"], "content": r["content"]}
            tc = r["tool_calls"]
            if tc:
                m["tool_calls"] = tc if isinstance(tc, list) else json.loads(tc)
            if r["tool_call_id"]:
                m["tool_call_id"] = r["tool_call_id"]
            if r["name"]:
                m["name"] = r["name"]
            out.append(m)
        return out

    async def add_note(self, uid: str, content: str) -> int:
        """Save a note and return its id."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                "INSERT INTO notes (user_id,content) VALUES ($1,$2) RETURNING id",
                uid, content)
        return row["id"]

    async def list_notes(self, uid: str, limit: int = 50) -> list[dict]:
        """List a user's notes, newest first."""
        async with self.pool.acquire() as c:
            rows = await c.fetch(
                """SELECT id,content,created_at FROM notes
                   WHERE user_id=$1 ORDER BY id DESC LIMIT $2""", uid, limit)
        return [dict(r) for r in rows]

    async def record_code_run(self, uid: str, provider: str, language: str, code: str,
                              stdout: Optional[str], stderr: Optional[str],
                              exit_code: Optional[int]) -> None:
        """Record a sandbox execution with truncated output."""
        async with self.pool.acquire() as c:
            await c.execute(
                """INSERT INTO code_runs (user_id,provider,language,code,
                   stdout,stderr,exit_code) VALUES ($1,$2,$3,$4,$5,$6,$7)""",
                uid, provider, language, code[:20000],
                (stdout or "")[:20000], (stderr or "")[:20000], exit_code)

    async def create_session(self, uid: str, ip: Optional[str],
                             user_agent: Optional[str], ttl_days: int) -> str:
        """Create a session row and return its id."""
        sid = uuid.uuid4().hex
        async with self.pool.acquire() as c:
            await c.execute(
                """INSERT INTO sessions (id,user_id,expires_at,ip,user_agent)
                   VALUES ($1,$2,NOW()+($3||' days')::interval,$4,$5)""",
                sid, uid, str(ttl_days), ip, (user_agent or "")[:300])
        return sid

    async def get_session(self, sid: str) -> Optional[dict]:
        """Fetch a session by id."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                "SELECT id,user_id,expires_at,revoked FROM sessions WHERE id=$1", sid)
        return dict(row) if row else None

    async def touch_session(self, sid: str) -> None:
        """Update a session's last-used timestamp."""
        async with self.pool.acquire() as c:
            await c.execute("UPDATE sessions SET last_used_at=NOW() WHERE id=$1", sid)

    async def revoke_session(self, sid: str) -> None:
        """Revoke one session."""
        async with self.pool.acquire() as c:
            await c.execute("UPDATE sessions SET revoked=TRUE WHERE id=$1", sid)

    async def revoke_all_sessions(self, uid: str) -> int:
        """Revoke every live session for a user."""
        async with self.pool.acquire() as c:
            res = await c.execute(
                "UPDATE sessions SET revoked=TRUE WHERE user_id=$1 AND revoked=FALSE", uid)
        try:
            return int(res.split()[-1])
        except Exception:
            return 0

    async def store_refresh(self, sid: str, token_hash: str, ttl_days: int) -> None:
        """Store the hash of an issued refresh token."""
        async with self.pool.acquire() as c:
            await c.execute(
                """INSERT INTO refresh_tokens (session_id,token_hash,expires_at)
                   VALUES ($1,$2,NOW()+($3||' days')::interval)""",
                sid, token_hash, str(ttl_days))

    async def get_refresh(self, token_hash: str) -> Optional[dict]:
        """Look up a refresh token by its hash."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                """SELECT id,session_id,expires_at,used_at,replaced_by
                   FROM refresh_tokens WHERE token_hash=$1""", token_hash)
        return dict(row) if row else None

    async def mark_refresh_used(self, token_hash: str, new_hash: str) -> bool:
        """Atomically consume a refresh token; returns False if already consumed."""
        async with self.pool.acquire() as c:
            result = await c.execute(
                """UPDATE refresh_tokens SET used_at=NOW(),replaced_by=$1
                   WHERE token_hash=$2 AND used_at IS NULL""", new_hash, token_hash)
        return result.endswith("1")

    async def add_attachment(self, uid: str, filename: Optional[str], stored: str,
                             ctype: Optional[str], size: int) -> str:
        """Record an uploaded attachment and return its id."""
        aid = uuid.uuid4().hex
        async with self.pool.acquire() as c:
            await c.execute(
                """INSERT INTO attachments (id,user_id,filename,stored_name,
                   content_type,size_bytes) VALUES ($1,$2,$3,$4,$5,$6)""",
                aid, uid, filename, stored, ctype, size)
        return aid

    async def list_attachments(self, uid: str, limit: int = 100) -> list[dict]:
        """List a user's attachments, newest first."""
        async with self.pool.acquire() as c:
            rows = await c.fetch(
                """SELECT id,filename,stored_name,content_type,size_bytes,created_at
                   FROM attachments WHERE user_id=$1 ORDER BY created_at DESC LIMIT $2""",
                uid, limit)
        return [dict(r) for r in rows]

    async def get_attachment(self, aid: str, uid: str) -> Optional[dict]:
        """Fetch one attachment owned by a user."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                """SELECT id,filename,stored_name,content_type,size_bytes
                   FROM attachments WHERE id=$1 AND user_id=$2""", aid, uid)
        return dict(row) if row else None

    async def register_media(self, stored_name: str, uid: str, size: int,
                             content_type: Optional[str] = None,
                             original_name: Optional[str] = None) -> None:
        """Register an object so media access is scoped to its owner."""
        async with self.pool.acquire() as c:
            await c.execute(
                """INSERT INTO media_objects
                   (stored_name,user_id,content_type,size_bytes,original_name)
                   VALUES ($1,$2,$3,$4,$5)
                   ON CONFLICT (stored_name) DO UPDATE SET
                     size_bytes=EXCLUDED.size_bytes, content_type=EXCLUDED.content_type,
                     original_name=EXCLUDED.original_name""",
                stored_name, uid, content_type, size, original_name)

    async def get_media(self, stored_name: str, uid: str) -> Optional[dict]:
        """Fetch metadata for an object owned by the caller, including legacy uploads."""
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                """SELECT stored_name,user_id,content_type,size_bytes,original_name
                   FROM media_objects WHERE stored_name=$1 AND user_id=$2""",
                stored_name, uid)
            if row:
                return dict(row)
            legacy = await c.fetchrow(
                """SELECT stored_name,user_id,content_type,size_bytes,filename AS original_name
                   FROM attachments WHERE stored_name=$1 AND user_id=$2
                   LIMIT 1""", stored_name, uid)
        return dict(legacy) if legacy else None


db = Database(DATABASE_URL)


# ---------------- RATE LIMITER ----------------
class RateLimiter:
    """Fixed-window rate limiter backed by Redis, or memory when Redis is absent."""
    def __init__(self) -> None:
        """Set up the in-memory fallback buckets."""
        self.redis = None
        self._mem: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._last_prune = time.time()

    async def init(self) -> None:
        """Connect to Redis when REDIS_URL is configured."""
        if not REDIS_URL:
            return
        try:
            import redis.asyncio as aioredis
            self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await self.redis.ping()
            log.info("redis connected")
        except Exception as e:
            log.warning("redis unavailable: %s", e)

    async def close(self) -> None:
        """Close the Redis connection if one is open."""
        if self.redis:
            await self.redis.aclose()

    async def hit(self, key: str, limit: int, window: int = 60) -> None:
        """Count one request against a key, raising 429 past the limit."""
        if self.redis:
            k = f"rl:{key}:{window}"
            n = await self.redis.incr(k)
            if n == 1:
                await self.redis.expire(k, window)
            if n > limit:
                raise HTTPException(429, "rate limit exceeded")
            return
        now = time.time()
        if now - self._last_prune > 300:
            self._prune(now)
        dq = self._mem[f"{key}:{window}"]
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(429, "rate limit exceeded")
        dq.append(now)

    def _prune(self, now: float) -> None:
        """Drop in-memory buckets that have fully aged out."""
        self._last_prune = now
        for key in [k for k, dq in self._mem.items() if not dq or now - dq[-1] > 3600]:
            self._mem.pop(key, None)


rate_limiter = RateLimiter()


# ---------------- MEDIA STORE ----------------
class MediaStore:
    """Unified local/S3-compatible object store with a small local analysis cache."""
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache = MEDIA_CACHE_DIR
        self.s3 = None
        if MEDIA_STORAGE == "s3":
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError("boto3 is required when MEDIA_STORAGE=s3") from exc
            self.s3 = boto3.client(
                "s3",
                endpoint_url=S3_ENDPOINT_URL or None,
                region_name=S3_REGION or None,
                aws_access_key_id=S3_ACCESS_KEY_ID or None,
                aws_secret_access_key=S3_SECRET_ACCESS_KEY or None,
                config=Config(signature_version="s3v4", retries={"max_attempts": 4, "mode": "standard"}),
            )

    @staticmethod
    def _safe_key(key: str) -> str:
        key = str(key or "").replace("\\", "/").lstrip("/")
        parts = [p for p in key.split("/") if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts):
            raise ValueError("invalid media key")
        return "/".join(parts)

    def _key(self, ext: str) -> str:
        owner = re.sub(r"[^a-zA-Z0-9_-]", "", CURRENT_USER_ID.get())[:80] or "system"
        ext = (ext or "bin").lstrip(".").lower()[:10]
        name = f"{uuid.uuid4().hex}.{ext}"
        return f"{S3_PREFIX}/{owner}/{name}" if S3_PREFIX else f"{owner}/{name}"

    def _local_path(self, key: str) -> Path:
        # Strip the object-store prefix so local fallback remains portable.
        clean = self._safe_key(key)
        if S3_PREFIX and clean.startswith(S3_PREFIX + "/"):
            clean = clean[len(S3_PREFIX) + 1:]
        p = (self.root / clean).resolve()
        p.relative_to(self.root.resolve())
        return p

    def save_bytes(self, data: bytes, ext: str, content_type: Optional[str] = None,
                   original_name: Optional[str] = None) -> str:
        if len(data) > MEDIA_MAX_BYTES:
            raise ValueError(f"file exceeds {MEDIA_MAX_BYTES} bytes")
        key = self._key(ext)
        if MEDIA_STORAGE == "local":
            p = self._local_path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        else:
            ctype = content_type or mimetypes.guess_type(key)[0] or "application/octet-stream"
            self.s3.put_object(Bucket=S3_BUCKET, Key=self._safe_key(key), Body=data,
                               ContentType=ctype, CacheControl="private, max-age=31536000")
        return key

    def save_b64(self, b64_data: str, ext: str) -> str:
        return self.save_bytes(base64.b64decode(b64_data), ext)

    def path(self, filename: str) -> Optional[Path]:
        """Return a local path, materializing an S3 object into the analysis cache when needed."""
        try:
            key = self._safe_key(filename)
        except ValueError:
            return None
        if MEDIA_STORAGE == "local":
            try:
                p = self._local_path(key)
            except (OSError, ValueError):
                return None
            return p if p.exists() and p.is_file() else None
        cache_name = hashlib.sha256(key.encode()).hexdigest() + "_" + Path(key).name
        cached = self.cache / cache_name
        if cached.exists() and cached.is_file():
            return cached
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(S3_BUCKET, key, str(cached))
            return cached
        except Exception as exc:
            log.warning("media materialization failed for %s: %s", key, exc)
            try:
                cached.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def read(self, filename: str) -> Optional[bytes]:
        p = self.path(filename)
        return p.read_bytes() if p else None

    def delete(self, filename: str) -> None:
        try:
            key = self._safe_key(filename)
        except ValueError:
            return
        if MEDIA_STORAGE == "local":
            try:
                self._local_path(key).unlink(missing_ok=True)
            except OSError:
                pass
        else:
            try:
                self.s3.delete_object(Bucket=S3_BUCKET, Key=key)
            except Exception as exc:
                log.warning("media deletion failed for %s: %s", key, exc)


media_store = MediaStore(MEDIA_DIR)


async def store_media(data: bytes, ext: str, content_type: Optional[str] = None,
                      original_name: Optional[str] = None) -> str:
    """Persist media and register ownership metadata."""
    stored = await asyncio.to_thread(media_store.save_bytes, data, ext, content_type, original_name)
    try:
        await db.register_media(stored, CURRENT_USER_ID.get(), len(data), content_type, original_name)
    except Exception:
        # The object remains recoverable; a later cleanup job can remove unregistered objects.
        log.exception("failed to register media metadata for %s", stored)
    return stored


# ---------------- AUTH ----------------
_ph = PasswordHasher()
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DUMMY_HASH = _ph.hash(secrets.token_urlsafe(32))


def hash_password(pw: str) -> str:
    """Hash a plaintext password with Argon2id."""
    return _ph.hash(pw)


def verify_password(stored: str, pw: str) -> bool:
    """Check a plaintext password against a stored Argon2 hash."""
    try:
        return _ph.verify(stored, pw)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def make_access_token(uid: str, sid: str) -> str:
    """Mint a short-lived access JWT bound to a session id."""
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": uid, "sid": sid, "iat": now,
                       "exp": now + timedelta(minutes=ACCESS_TTL_MIN),
                       "jti": uuid.uuid4().hex},
                      JWT_SECRET, algorithm=JWT_ALG)


def decode_access(token: str) -> Optional[dict]:
    """Decode an access token, returning None when it is invalid."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None


def new_refresh_token() -> tuple[str, str]:
    """Return a fresh refresh token and its storage hash."""
    raw = secrets.token_urlsafe(64)
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def hash_refresh(raw: Any) -> str:
    """Hash a raw refresh token for storage and lookup."""
    return hashlib.sha256(raw.encode()).hexdigest()


def set_auth_cookies(resp: Response, access: str, refresh: str) -> None:
    """Write the access and refresh cookies onto a response."""
    common = dict(httponly=True, samesite="none" if CROSS_ORIGIN else "lax",
                  secure=CROSS_ORIGIN or IS_PROD, path="/")
    resp.set_cookie(COOKIE_ACCESS, access, max_age=ACCESS_TTL_MIN * 60, **common)
    resp.set_cookie(COOKIE_REFRESH, refresh, max_age=REFRESH_TTL_DAYS * 86400, **common)


def clear_auth_cookies(resp: Response) -> None:
    """Expire both auth cookies on a response."""
    for name in (COOKIE_ACCESS, COOKIE_REFRESH):
        resp.delete_cookie(name, path="/",
                           samesite="none" if CROSS_ORIGIN else "lax",
                           secure=CROSS_ORIGIN or IS_PROD)


async def _session_from_token(token: str) -> Optional[dict]:
    """Resolve a live, unrevoked session from an access token."""
    payload = decode_access(token)
    if not payload or not payload.get("sid"):
        return None
    sess = await db.get_session(payload["sid"])
    if not sess or sess["revoked"]:
        return None
    exp = sess["expires_at"]
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp < datetime.now(timezone.utc):
        return None
    return {"user_id": payload["sub"], "session_id": payload["sid"]}


async def get_current_user(request: Request,
                           authorization: Optional[str] = Header(default=None),
                           yiz_access: Optional[str] = Cookie(default=None)) -> dict:
    """Authenticate the caller, enforcing the CSRF header for cookie auth."""
    token, via_cookie = None, False
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif yiz_access:
        token, via_cookie = yiz_access, True
    if not token:
        raise HTTPException(401, "not authenticated")
    sess = await _session_from_token(token)
    if not sess:
        raise HTTPException(401, "invalid or expired session")
    if via_cookie and request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get("x-requested-with") != "yiz":
            raise HTTPException(403, "missing CSRF header")
    user = await db.get_user(sess["user_id"])
    if not user or not user["is_active"]:
        raise HTTPException(401, "user not found")
    await db.touch_session(sess["session_id"])
    return {"id": user["id"], "email": user["email"],
            "name": user["name"], "session_id": sess["session_id"]}


async def _issue_session(user: dict, request: Request, resp: Response) -> dict:
    """Create a session, set its cookies, and return the public user record."""
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    sid = await db.create_session(user["id"], ip, ua, REFRESH_TTL_DAYS)
    access = make_access_token(user["id"], sid)
    raw, hashed = new_refresh_token()
    await db.store_refresh(sid, hashed, REFRESH_TTL_DAYS)
    set_auth_cookies(resp, access, raw)
    return {"id": user["id"], "email": user["email"], "name": user["name"]}


# ---------------- OAUTH ----------------
class OAuthProvider:
    """Base class for the supported OAuth identity providers."""
    name = ""
    authorize_url = ""
    token_url = ""
    scope = ""

    def __init__(self, cid: str, secret: str) -> None:
        """Store the provider's client credentials."""
        self.client_id, self.client_secret = cid, secret

    @property
    def enabled(self) -> bool:
        """Whether this provider has everything it needs to run."""
        return bool(self.client_id and self.client_secret and OAUTH_REDIRECT_BASE)

    def redirect_uri(self) -> str:
        """The callback URL registered with the provider."""
        return f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/{self.name}/callback"

    def authorize(self, state: str) -> str:
        """Build the provider's authorization URL."""
        raise NotImplementedError
    async def exchange(self, code: str) -> dict:
        """Exchange an authorization code for tokens."""
        raise NotImplementedError
    async def fetch_identity(self, tokens: dict) -> dict:
        """Fetch the verified identity behind a token set."""
        raise NotImplementedError


class GoogleOAuth(OAuthProvider):
    """Google OpenID Connect provider."""
    name = "google"
    authorize_url = "https://accounts.google.com/o/oauth2/v2/auth"
    token_url = "https://oauth2.googleapis.com/token"
    scope = "openid email profile"

    def authorize(self, state: str) -> str:
        """Build Google's authorization URL."""
        return self.authorize_url + "?" + urlencode({
            "client_id": self.client_id, "redirect_uri": self.redirect_uri(),
            "response_type": "code", "scope": self.scope, "state": state,
            "access_type": "online", "prompt": "select_account"})

    async def exchange(self, code: str) -> dict:
        """Exchange a Google authorization code for tokens."""
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(self.token_url, data={
                "code": code, "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri(),
                "grant_type": "authorization_code"})
        r.raise_for_status()
        return r.json()

    async def fetch_identity(self, tokens: dict) -> dict:
        """Fetch a Google identity, requiring a verified email."""
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://openidconnect.googleapis.com/v1/userinfo",
                            headers={"Authorization": f"Bearer {tokens['access_token']}"})
        r.raise_for_status()
        d = r.json()
        if not d.get("email") or not d.get("email_verified"):
            raise HTTPException(400, "google account has no verified email")
        return {"sub": str(d["sub"]), "email": d["email"], "name": d.get("name")}


class GitHubOAuth(OAuthProvider):
    """GitHub OAuth provider."""
    name = "github"
    authorize_url = "https://github.com/login/oauth/authorize"
    token_url = "https://github.com/login/oauth/access_token"
    scope = "read:user user:email"

    def authorize(self, state: str) -> str:
        """Build GitHub's authorization URL."""
        return self.authorize_url + "?" + urlencode({
            "client_id": self.client_id, "redirect_uri": self.redirect_uri(),
            "scope": self.scope, "state": state, "allow_signup": "false"})

    async def exchange(self, code: str) -> dict:
        """Exchange a GitHub authorization code for tokens."""
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(self.token_url, data={
                "code": code, "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri()},
                headers={"Accept": "application/json"})
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            raise HTTPException(400, f"github: {d.get('error_description', d['error'])}")
        return d

    async def fetch_identity(self, tokens: dict) -> dict:
        """Fetch a GitHub identity, requiring a verified primary email."""
        headers = {"Authorization": f"Bearer {tokens['access_token']}",
                   "Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(timeout=20) as c:
            u = await c.get("https://api.github.com/user", headers=headers)
            u.raise_for_status()
            user = u.json()
            email = user.get("email")
            if not email:
                e = await c.get("https://api.github.com/user/emails", headers=headers)
                e.raise_for_status()
                primary = next((x for x in e.json()
                                if x.get("primary") and x.get("verified")), None)
                email = primary["email"] if primary else None
        if not email:
            raise HTTPException(400, "github account has no verified primary email")
        return {"sub": str(user["id"]), "email": email,
                "name": user.get("name") or user.get("login")}


oauth_providers = {}
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth_providers["google"] = GoogleOAuth(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth_providers["github"] = GitHubOAuth(GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET)


def _sign_state(provider: str, next_url: Optional[str]) -> str:
    """Sign a short-lived OAuth state token."""
    return jwt.encode({"p": provider, "n": next_url, "r": secrets.token_urlsafe(16),
                       "exp": datetime.now(timezone.utc) + timedelta(minutes=10)},
                      JWT_SECRET, algorithm=JWT_ALG)


def _verify_state(state: str) -> dict:
    """Verify an OAuth state token, rejecting anything tampered with."""
    try:
        return jwt.decode(state, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        raise HTTPException(400, "invalid oauth state")


def _safe_next(next_url: Optional[str]) -> str:
    """Constrain a post-login redirect to a same-site path."""
    if next_url and next_url.startswith("/") and not next_url.startswith("//"):
        return next_url
    return FRONTEND_URL or "/"


async def _upsert_oauth_user(provider: str, ident: dict) -> dict:
    """Find, link, or create the local user behind an OAuth identity."""
    user = await db.get_user_by_oauth(provider, ident["sub"])
    if user:
        return user
    user = await db.get_user_by_email(ident["email"])
    if user:
        await db.link_oauth(user["id"], provider, ident["sub"])
        return await db.get_user(user["id"])
    return await db.create_user(ident["email"], None, ident.get("name"),
                                oauth_provider=provider, oauth_sub=ident["sub"])


# ---------------- N8N ----------------
class N8NClient:
    """Webhook client for triggering n8n workflows."""
    @property
    def enabled(self) -> bool:
        """Whether an n8n webhook base URL is configured."""
        return bool(N8N_WEBHOOK_URL)

    async def trigger(self, workflow: str, payload: Optional[dict]) -> dict[str, Any]:
        """POST a payload to an n8n workflow webhook."""
        if not self.enabled:
            return {"error": "n8n not configured (set N8N_WEBHOOK_URL)"}
        url = f"{N8N_WEBHOOK_URL}/{workflow.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        if N8N_API_KEY:
            headers["X-N8N-API-KEY"] = N8N_API_KEY
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(url, json=payload, headers=headers)
            return {"status": r.status_code, "workflow": workflow,
                    "response": r.text[:2000]}
        except httpx.HTTPError as e:
            return {"error": f"n8n call failed: {e}"}


n8n_client = N8NClient()


# ---------------- MCP ----------------
def _validate_mcp_url(name: str, raw: str) -> Optional[str]:
    """Return a safe absolute MCP base URL, or None when the entry is rejected."""
    url = (raw or "").strip().rstrip("/")
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError as e:
        log.warning("MCP server %s rejected: malformed URL (%s)", name, e)
        return None
    if parsed.scheme not in ("http", "https"):
        log.warning("MCP server %s rejected: scheme %r not allowed", name, parsed.scheme)
        return None
    if not parsed.hostname:
        log.warning("MCP server %s rejected: no hostname", name)
        return None
    if IS_PROD and parsed.scheme != "https" and not MCP_ALLOW_PRIVATE:
        log.warning("MCP server %s rejected: https required in production", name)
        return None
    if not MCP_ALLOW_PRIVATE:
        host = parsed.hostname.lower()
        if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
            log.warning("MCP server %s rejected: localhost not allowed", name)
            return None
        addresses = []
        try:
            addresses.append(ipaddress.ip_address(host))
        except ValueError:
            try:
                infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                           proto=socket.IPPROTO_TCP)
                addresses = [ipaddress.ip_address(i[4][0]) for i in infos]
            except (socket.gaierror, ValueError) as e:
                log.warning("MCP server %s rejected: cannot resolve %s (%s)", name, host, e)
                return None
        for ip in addresses:
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                log.warning("MCP server %s rejected: %s resolves to non-public %s", name, host, ip)
                return None
    return url


def _normalise_mcp_tool(t: dict) -> dict:
    """Coerce one advertised MCP tool into a schema a provider will accept."""
    schema = t.get("inputSchema")
    if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
        schema = {"type": "object", "properties": {}}
    else:
        schema = dict(schema)
        schema["type"] = "object"
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = {}
        schema["properties"] = {k: v for k, v in props.items() if isinstance(v, dict)}
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [r for r in required
                                  if isinstance(r, str) and r in schema["properties"]]
        else:
            schema.pop("required", None)
    description = t.get("description")
    return {"name": t["name"],
            "description": description if isinstance(description, str) else "",
            "inputSchema": schema}


class MCPClient:
    """Minimal JSON-RPC client for configured MCP servers."""
    def __init__(self) -> None:
        """Parse and validate the configured name=url server list."""
        self.servers = {}
        self.rejected = {}
        for entry in MCP_SERVERS.split(","):
            if "=" not in entry:
                continue
            name, url = entry.split("=", 1)
            name = name.strip()
            if not name:
                continue
            safe = _validate_mcp_url(name, url)
            if safe:
                self.servers[name] = safe
            else:
                self.rejected[name] = url.strip()
        self._cache = {}

    def list_servers(self) -> list[str]:
        """Names of the configured MCP servers."""
        return list(self.servers.keys())

    async def _rpc(self, server: str, method: str, params: dict) -> dict[str, Any]:
        """Send one JSON-RPC call to a configured server."""
        if server not in self.servers:
            if server in self.rejected:
                return {"error": f"MCP server {server!r} was rejected by URL validation"}
            return {"error": f"unknown MCP server: {server}"}
        url = f"{self.servers[server]}/mcp"
        payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex,
                   "method": method, "params": params}
        try:
            async with httpx.AsyncClient(timeout=MCP_TIMEOUT, follow_redirects=False) as c:
                r = await c.post(url, json=payload,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
                r.raise_for_status()
                body = await r.aread()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.text[:300]
            except Exception:
                pass
            log.warning("MCP %s %s -> HTTP %s", server, method, e.response.status_code)
            return {"error": f"MCP server returned HTTP {e.response.status_code}",
                    "detail": detail}
        except httpx.HTTPError as e:
            return {"error": f"MCP call failed: {e}"}

        if len(body) > MCP_MAX_RESPONSE_BYTES:
            return {"error": f"MCP response too large ({len(body)} bytes)"}
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {"error": f"MCP response was not valid JSON: {e}"}
        if not isinstance(data, dict):
            return {"error": "MCP response was not a JSON-RPC object"}
        if isinstance(data.get("error"), dict):
            err = data["error"]
            return {"error": f"MCP error {err.get('code')}: {err.get('message')}"}
        return data

    async def list_tools(self, server: str) -> list[dict]:
        """List a server's tools, cached per process."""
        if server in self._cache:
            return self._cache[server]
        resp = await self._rpc(server, "tools/list", {})
        if "error" in resp:
            log.warning("MCP tools/list failed for %s: %s", server, resp["error"])
            return []
        tools = (resp.get("result") or {}).get("tools", []) or []
        tools = [_normalise_mcp_tool(t) for t in tools
                 if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"]]
        self._cache[server] = tools
        return tools

    async def call_tool(self, server: str, tool: str, args: Optional[dict]) -> dict[str, Any]:
        """Invoke a tool on a server."""
        resp = await self._rpc(server, "tools/call",
                               {"name": tool, "arguments": args})
        if "error" in resp:
            return resp
        return resp.get("result", {})

    async def all_tools(self) -> list[dict]:
        """Every MCP tool across servers, namespaced for the model."""
        out = []
        for server in self.servers:
            for t in await self.list_tools(server):
                t = _normalise_mcp_tool(t)
                out.append({"server": server,
                            "name": f"mcp__{server}__{t['name']}",
                            "description": t["description"],
                            "input_schema": t["inputSchema"]})
        return out


mcp_client = MCPClient()


# ---------------- PLUGINS ----------------
class PluginLoader:
    """Loads optional tool plugins from PLUGINS_DIR."""
    def __init__(self, directory: Path) -> None:
        """Store the plugin directory."""
        self.directory = directory
        self.loaded = []

    def load_all(self, reserved_names: frozenset[str] = frozenset()) -> list[dict]:
        """Import every well-formed plugin module in the directory.

        reserved_names is normally the set of already-registered built-in
        tool names, so a plugin cannot silently shadow one of them.
        """
        self.loaded = []
        if not self.directory.exists():
            return self.loaded
        for path in sorted(self.directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not all(hasattr(mod, a) for a in
                           ("NAME", "DESCRIPTION", "PARAMETERS", "run")):
                    log.warning("plugin %s: missing required attrs", path.name)
                    continue
                if (not isinstance(mod.NAME, str)
                        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", mod.NAME)
                        or not isinstance(mod.DESCRIPTION, str)
                        or not isinstance(mod.PARAMETERS, dict)
                        or not callable(mod.run)):
                    log.warning("plugin %s: invalid metadata (name/description/"
                               "parameters/run)", path.name)
                    continue
                if mod.NAME in reserved_names or any(mod.NAME == p["name"]
                                                      for p in self.loaded):
                    log.warning("plugin %s: name %r collides with a built-in "
                               "tool or another plugin, skipping", path.name, mod.NAME)
                    continue
                self.loaded.append({"name": mod.NAME, "description": mod.DESCRIPTION,
                                    "parameters": mod.PARAMETERS, "run": mod.run,
                                    "source": path.name})
                log.info("plugin loaded: %s (%s)", mod.NAME, path.name)
            except Exception as e:
                log.warning("plugin %s failed: %s", path.name, e)
        return self.loaded


plugin_loader = PluginLoader(PLUGINS_DIR)


# ---------------- SANDBOX ----------------
_ALIASES = {"py": "python", "python3": "python", "js": "javascript",
            "node": "javascript", "ts": "typescript", "rb": "ruby",
            "sh": "bash", "shell": "bash", "c++": "cpp", "cs": "csharp",
            "golang": "go", "rs": "rust", "psql": "sql", "postgres": "sql"}
_EXT = {"python": "py", "javascript": "js", "typescript": "ts", "go": "go",
        "rust": "rs", "java": "java", "c": "c", "cpp": "cpp", "csharp": "cs",
        "bash": "sh", "ruby": "rb", "php": "php", "sql": "sql",
        "kotlin": "kt", "swift": "swift", "lua": "lua", "r": "r", "perl": "pl"}


class SandboxError(Exception):
    """Raised when the code sandbox cannot run a submission."""
    pass


class Sandbox:
    """Remote code execution through Piston or E2B."""
    def __init__(self) -> None:
        """Select the configured execution provider."""
        self.provider = SANDBOX_PROVIDER
        self._runtimes = {}
        self._loaded = False

    async def execute(self, language: str, code: str, stdin: Optional[str],
                      user_id: str) -> dict[str, Any]:
        """Run code through the configured provider, rate limited per user."""
        if self.provider == "off":
            return {"error": "code sandbox disabled"}
        await rate_limiter.hit(f"sbx:{user_id}", SANDBOX_RUNS_PER_MIN, 60)
        if self.provider == "piston":
            return await self._piston(language, code, stdin)
        if self.provider == "e2b":
            return await self._e2b(language, code, stdin)
        return {"error": f"unknown provider: {self.provider}"}

    async def _load_runtimes(self) -> None:
        """Cache the newest Piston runtime for each language."""
        if self._loaded:
            return
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{PISTON_URL}/runtimes")
        r.raise_for_status()
        by_lang = defaultdict(list)
        for rt in r.json():
            by_lang[rt["language"]].append(rt)
        for lang, items in by_lang.items():
            items.sort(key=lambda x: x.get("version", ""), reverse=True)
            self._runtimes[lang] = items[0]
        self._loaded = True

    def _resolve(self, language: str) -> Optional[str]:
        """Map a language alias onto an available runtime."""
        lang = (language or "").strip().lower()
        lang = _ALIASES.get(lang, lang)
        return lang if lang in self._runtimes else None

    async def _piston(self, language: str, code: str, stdin: Optional[str]) -> dict[str, Any]:
        """Execute code on Piston."""
        await self._load_runtimes()
        lang = self._resolve(language)
        if not lang:
            return {"error": f"unsupported language: {language}",
                    "available": sorted(self._runtimes.keys())}
        rt = self._runtimes[lang]
        payload = {"language": rt["language"], "version": rt["version"],
                   "files": [{"name": f"main.{_EXT.get(lang, 'txt')}", "content": code}],
                   "stdin": stdin or ""}
        try:
            async with httpx.AsyncClient(timeout=SANDBOX_TIMEOUT) as c:
                r = await c.post(f"{PISTON_URL}/execute", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            return {"error": f"piston failed: {e}"}
        data = r.json()
        run = data.get("run") or {}
        return {"provider": "piston", "language": rt["language"],
                "version": rt["version"], "stdout": run.get("stdout", ""),
                "stderr": run.get("stderr", ""), "exit_code": run.get("code"),
                "output": run.get("output", "")}

    async def _e2b(self, language: str, code: str, stdin: Optional[str]) -> dict[str, Any]:
        """Execute code in an E2B sandbox."""
        if not E2B_API_KEY:
            return {"error": "E2B_API_KEY not set"}
        try:
            from e2b_code_interpreter import Sandbox as E2BSandbox
        except ImportError:
            return {"error": "e2b-code-interpreter not installed"}
        lang = (language or "").strip().lower()

        def _run() -> dict[str, Any]:
            """Run the submission inside a disposable E2B sandbox."""
            sbx = E2BSandbox(api_key=E2B_API_KEY)
            try:
                if lang in ("python", "py", "python3"):
                    r = sbx.run_code(code)
                elif lang in ("javascript", "js", "node", "typescript", "ts"):
                    r = sbx.run_code(code, language="js")
                else:
                    sbx.files.write(f"/tmp/main.{_EXT.get(lang, 'txt')}", code)
                    r = sbx.commands.run("true")
                return {"stdout": "\n".join(r.logs.stdout),
                        "stderr": "\n".join(r.logs.stderr),
                        "error": str(r.error) if r.error else None}
            finally:
                try:
                    sbx.kill()
                except Exception:
                    pass

        try:
            result = await asyncio.wait_for(asyncio.to_thread(_run),
                                            timeout=SANDBOX_TIMEOUT)
        except asyncio.TimeoutError:
            return {"error": f"e2b timed out after {SANDBOX_TIMEOUT}s"}
        return {"provider": "e2b", "language": lang,
                "stdout": result["stdout"],
                "stderr": result["stderr"] + (("\n" + result["error"]) if result["error"] else ""),
                "exit_code": 1 if result["error"] else 0}


sandbox = Sandbox()


# ---------------- LLM ----------------
class LLMError(Exception):
    """Raised when no provider is configured or a provider call fails."""
    pass


_PROVIDERS: dict[str, tuple[bool, str]] = {
    "anthropic": (bool(ANTHROPIC_API_KEY), ANTHROPIC_MODEL),
    "openai": (bool(OPENAI_API_KEY), OPENAI_MODEL),
    "kimi": (bool(KIMI_API_KEY), KIMI_MODEL),
    "deepseek": (bool(DEEPSEEK_API_KEY), DEEPSEEK_MODEL),
    "ollama": (bool(OLLAMA_BASE_URL or OLLAMA_API_KEY), OLLAMA_MODEL),
}
# Order "auto" tries providers in, and the order /api/models lists them in.
# Kimi and DeepSeek sit ahead of Ollama, so a locally/cloud-hosted Ollama model
# is only used when neither hosted provider has credentials configured.
_PROVIDER_ORDER = ("anthropic", "openai", "kimi", "deepseek", "ollama")


def configured_providers() -> list[str]:
    """Names of every provider that has credentials in this environment."""
    return [n for n in _PROVIDER_ORDER if _PROVIDERS[n][0]]


def pick_provider() -> tuple[str, str]:
    """Choose the default provider: LLM_PROVIDER if set, else anthropic > openai > ollama."""
    if LLM_PROVIDER:
        if LLM_PROVIDER not in _PROVIDERS:
            raise LLMError(f"LLM_PROVIDER={LLM_PROVIDER!r} is not a known provider.")
        configured, model = _PROVIDERS[LLM_PROVIDER]
        if not configured:
            raise LLMError(f"LLM_PROVIDER={LLM_PROVIDER!r} has no credentials set.")
        return LLM_PROVIDER, model
    for name in _PROVIDER_ORDER:
        configured, model = _PROVIDERS[name]
        if configured:
            return name, model
    raise LLMError("No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY "
                   "or OLLAMA_API_KEY / OLLAMA_BASE_URL.")


def _provider_for_model(model: str) -> Optional[str]:
    """Guess which configured provider serves a bare model id."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt-", "gpt4", "o1", "o3", "o4", "chatgpt", "text-davinci")):
        return "openai"
    if m.startswith("kimi"):
        return "kimi"
    if m.startswith("deepseek"):
        return "deepseek"
    return None


def resolve_model(requested: Optional[str]) -> tuple[str, str]:
    """Resolve a caller-supplied model into (provider, model).

    Accepts "provider:model" (e.g. "openai:gpt-4o"), a bare provider name
    ("anthropic"), or a bare model id ("claude-sonnet-4-20250514").
    Falls back to the environment default when nothing is requested.
    """
    requested = (requested or "").strip()
    if not requested:
        return pick_provider()
    if not ALLOW_MODEL_OVERRIDE:
        return pick_provider()

    provider, model = None, requested
    if ":" in requested:
        head, tail = requested.split(":", 1)
        if head.strip().lower() in _PROVIDERS:
            provider, model = head.strip().lower(), tail.strip()
    elif requested.lower() in _PROVIDERS:
        provider, model = requested.lower(), ""

    if provider is None:
        provider = _provider_for_model(model)
    if provider is None:
        # Bare, unrecognised model id. Only route it to Ollama when Ollama is
        # the explicitly chosen provider, since Ollama model names are arbitrary.
        if LLM_PROVIDER == "ollama" and _PROVIDERS["ollama"][0]:
            provider = "ollama"
        else:
            raise LLMError(
                f"Unknown model {model!r}. Use 'provider:model' (e.g. "
                "'ollama:llama3.1') or pick one from /api/models.")

    configured, default_model = _PROVIDERS[provider]
    if not configured:
        raise LLMError(f"provider {provider!r} has no credentials set")
    return provider, (model or default_model)


_model_cache: dict[str, tuple[float, list[str]]] = {}


async def _list_provider_models(provider: str) -> list[str]:
    """Live model ids for one configured provider, cached for MODEL_LIST_TTL."""
    hit = _model_cache.get(provider)
    if hit and time.time() - hit[0] < MODEL_LIST_TTL:
        return hit[1]

    models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            if provider == "anthropic":
                r = await c.get(f"{ANTHROPIC_BASE_URL}/models?limit=100",
                                headers={"x-api-key": ANTHROPIC_API_KEY,
                                         "anthropic-version": ANTHROPIC_VERSION})
                r.raise_for_status()
                models = [m["id"] for m in r.json().get("data", []) if m.get("id")]
            elif provider == "openai":
                r = await c.get(f"{OPENAI_BASE_URL}/models",
                                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
                r.raise_for_status()
                models = [m["id"] for m in r.json().get("data", []) if m.get("id")]
                models = [m for m in models
                          if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))]
            elif provider in ("kimi", "deepseek"):
                base, key = _OPENAI_COMPAT[provider]
                r = await c.get(f"{base}/models",
                                headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                models = [m["id"] for m in r.json().get("data", []) if m.get("id")]
            elif provider == "ollama":
                base = OLLAMA_BASE_URL or "https://ollama.com/v1"
                headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
                r = await c.get(f"{base}/models", headers=headers)
                if r.status_code == 404 and base.endswith("/v1"):
                    r = await c.get(f"{base[:-3].rstrip('/')}/api/tags", headers=headers)
                    r.raise_for_status()
                    models = [m["name"] for m in r.json().get("models", []) if m.get("name")]
                else:
                    r.raise_for_status()
                    models = [m["id"] for m in r.json().get("data", []) if m.get("id")]
    except Exception as e:                      # never let a listing failure break the UI
        log.warning("model listing failed for %s: %s", provider, e)
        default = _PROVIDERS[provider][1]
        models = [default] if default else []

    models = sorted(dict.fromkeys(models))
    _model_cache[provider] = (time.time(), models)
    return models


async def _validate_model_override(provider: str, model: str) -> None:
    """Reject a caller-chosen model that isn't in that provider's live catalog.

    Fails open when the catalog call itself failed (empty list), so a flaky
    provider listing endpoint degrades to "unchecked", not "broken".
    """
    if not MODEL_OVERRIDE_STRICT:
        return
    available = await _list_provider_models(provider)
    if available and model not in available:
        raise LLMError(
            f"{model!r} is not in {provider}'s current model list. "
            "Choose one from /api/models.")


async def list_all_models() -> dict[str, Any]:
    """Every model available across every configured provider."""
    names = configured_providers()
    results = await asyncio.gather(*(_list_provider_models(n) for n in names),
                                   return_exceptions=True)
    out, flat = {}, []
    for name, res in zip(names, results):
        ids = res if isinstance(res, list) else []
        out[name] = ids
        flat += [f"{name}:{m}" for m in ids]
    try:
        default_provider, default_model = pick_provider()
        default = f"{default_provider}:{default_model}"
    except LLMError:
        default = None
    return {"providers": out, "models": flat, "default": default}


def _parse_tool_args(raw: str, tool_name: str) -> dict:
    """Parse streamed tool arguments, recovering truncated JSON where possible."""
    if not raw or raw.strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass
    for cut in range(len(raw) - 1, 0, -1):
        if raw[cut] not in "}]":
            continue
        head = raw[:cut + 1]
        for candidate in (head, head + _closers(head)):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return parsed if isinstance(parsed, dict) else {}
    try:
        parsed = json.loads(raw + _closers(raw))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    log.warning("unparseable tool args for %s (%d chars)", tool_name, len(raw))
    return {}


def _closers(fragment: str) -> str:
    """Closing brackets needed to balance a truncated JSON fragment."""
    stack, in_string, escaped = [], False, False
    for ch in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return ('"' if in_string else "") + "".join(reversed(stack))


def _to_openai(msgs: list[dict]) -> list[dict]:
    """Convert internal messages to the OpenAI chat format."""
    out = []
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{"id": tc["id"], "type": "function",
                                        "function": {"name": tc["name"],
                                                     "arguments": json.dumps(tc["args"])}}
                                       for tc in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _to_anthropic(msgs: list[dict]) -> tuple[str, list[dict]]:
    """Split internal messages into an Anthropic system prompt and turns."""
    system, out = "", []
    for m in msgs:
        if m["role"] == "system":
            system = m["content"]
            continue
        if m["role"] == "assistant" and m.get("tool_calls"):
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": tc["name"], "input": tc["args"]})
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": m["tool_call_id"],
                "content": m["content"]}]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return system, out


def _anthropic_tools(schemas: list[dict]) -> list[dict]:
    """Convert tool schemas to Anthropic's tool format."""
    return [{"name": s["name"], "description": s["description"],
             "input_schema": s["parameters"]} for s in schemas]


async def _stream_openai(msgs: list[dict], tools_list: list[dict], model: str,
                         provider: str = "openai") -> AsyncGenerator[dict, None]:
    """Stream an OpenAI-compatible completion (openai/kimi/deepseek/ollama)."""
    if provider not in _OPENAI_COMPAT:
        raise LLMError(f"{provider!r} is not an OpenAI-compatible provider")
    base, key = _OPENAI_COMPAT[provider]
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": model, "messages": _to_openai(msgs), "stream": True}
    if tools_list:
        payload["tools"] = [{"type": "function", "function": s} for s in tools_list]
        payload["tool_choice"] = "auto"
    acc = {}
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{base}/chat/completions",
                            headers=headers, json=payload) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")
                rid = uuid.uuid4().hex[:8]
                log.error("llm:%s %s %s: %s", rid, provider, r.status_code, body[:800])
                raise LLMError(f"LLM provider error (ref {rid})")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                except json.JSONDecodeError:
                    continue
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        yield {"type": "text", "delta": delta["content"]}
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
    if acc:
        calls = []
        for i in sorted(acc):
            s = acc[i]
            args = _parse_tool_args(s["args"], s["name"])
            calls.append({"id": s["id"] or f"call_{i}",
                          "name": s["name"], "args": args})
        yield {"type": "tool_calls", "calls": calls}


async def _stream_anthropic(msgs: list[dict], tools_list: list[dict],
                            model: str) -> AsyncGenerator[dict, None]:
    """Stream an Anthropic completion, yielding text and tool calls."""
    system, out = _to_anthropic(msgs)
    payload = {"model": model, "max_tokens": ANTHROPIC_MAX_TOKENS,
               "messages": out, "stream": True}
    if system:
        payload["system"] = system
    if tools_list:
        payload["tools"] = _anthropic_tools(tools_list)
    headers = {"x-api-key": ANTHROPIC_API_KEY,
               "anthropic-version": ANTHROPIC_VERSION,
               "Content-Type": "application/json"}
    blocks = {}
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{ANTHROPIC_BASE_URL}/messages",
                            headers=headers, json=payload) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")
                rid = uuid.uuid4().hex[:8]
                log.error("llm:%s anthropic %s: %s", rid, r.status_code, body[:800])
                raise LLMError(f"LLM provider error (ref {rid})")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "content_block_start":
                    blk = ev["content_block"]
                    if blk["type"] == "tool_use":
                        blocks[ev["index"]] = {"id": blk["id"], "name": blk["name"],
                                               "args_json": ""}
                elif t == "content_block_delta":
                    d = ev["delta"]
                    if d.get("type") == "text_delta":
                        yield {"type": "text", "delta": d["text"]}
                    elif d.get("type") == "input_json_delta":
                        idx = ev["index"]
                        if idx in blocks:
                            blocks[idx]["args_json"] += d.get("partial_json", "")
                elif t == "message_stop":
                    break
    if blocks:
        calls = []
        for i in sorted(blocks):
            b = blocks[i]
            args = _parse_tool_args(b["args_json"], b["name"])
            calls.append({"id": b["id"], "name": b["name"], "args": args})
        yield {"type": "tool_calls", "calls": calls}


# ---------------- PER-TASK MODEL ROUTING (kimi / deepseek) ----------------
# Anthropic and OpenAI models are picked explicitly via *_MODEL env vars, since
# their naming isn't consistent enough to route on safely. Kimi and DeepSeek
# each publish a small, predictable family (a general chat model plus a
# reasoning/"thinking" variant), so for those two providers we pick the
# specific model from the *live* catalog based on what the prompt looks like,
# rather than always using the single configured default.
#
# This is a lightweight heuristic, not a benchmarked router: it looks at
# surface features of the prompt (code fences, math/proof language, length)
# and matches them against model-name keywords returned by /api/models right
# now, so it keeps working if either provider renames or adds models. It will
# never be as good as picking per-task based on your own eval results.
AUTO_ROUTE_PROVIDERS = frozenset({"kimi", "deepseek"})

_CODE_RE = re.compile(
    r"```|\b(?:function|debug|traceback|stack trace|compile|refactor|"
    r"regex|unit test|class \w+|def \w+|import \w+)\b", re.I)
_REASONING_RE = re.compile(
    r"\b(?:prove|proof|step by step|derive|theorem|reasoning|logic puzzle|"
    r"solve for|optimi[sz]e|algorithm complexity)\b", re.I)

# Keyword to look for in a live model id for each (provider, task) pair.
# Empty string means "use the provider's own configured default".
_ROUTE_KEYWORDS: dict[str, dict[str, str]] = {
    "deepseek": {"code": "reasoner", "reasoning": "reasoner", "general": ""},
    "kimi": {"code": "k2", "reasoning": "thinking", "general": ""},
}


def _classify_task(prompt: str) -> str:
    """Rough (code | reasoning | general) label for routing purposes only."""
    if _CODE_RE.search(prompt or ""):
        return "code"
    if _REASONING_RE.search(prompt or "") or len(prompt or "") > 2000:
        return "reasoning"
    return "general"


async def _auto_pick_model(provider: str, prompt: str) -> Optional[str]:
    """Best-effort per-task model choice within one provider's own catalog.

    Returns None (meaning: use the configured default) if the provider isn't
    auto-routed, the keyword has no match in the live catalog, or the catalog
    call itself fails — routing degrades to the static default, it never
    blocks or errors the chat turn.
    """
    hints = _ROUTE_KEYWORDS.get(provider)
    if not hints:
        return None
    keyword = hints.get(_classify_task(prompt), "")
    if not keyword:
        return None
    available = await _list_provider_models(provider)
    matches = sorted((m for m in available if keyword in m.lower()), key=len)
    return matches[0] if matches else None


def _opaque_llm_error(e: Exception, provider: str) -> LLMError:
    """Wrap any exception as an LLMError with a logged reference id.

    Keeps raw provider error bodies (which can include account/billing
    detail) out of the user-facing message while still letting an operator
    find the real cause by grepping server logs for the ref id shown to the
    user.
    """
    if isinstance(e, LLMError):
        return e
    rid = uuid.uuid4().hex[:8]
    log.error("llm:%s %s unexpected error: %s: %s", rid, provider, type(e).__name__, e)
    return LLMError(f"LLM provider error (ref {rid})")


async def stream_chat(msgs: list[dict], tools_list: list[dict],
                      requested_model: Optional[str] = None,
                      routing_prompt: Optional[str] = None) -> AsyncGenerator[dict, None]:
    """Stream a completion, failing over to the next configured provider.

    A caller-pinned model (requested_model) is honored exactly — no silent
    failover, since the person asked for that model specifically. Otherwise
    every configured provider in _PROVIDER_ORDER is tried in turn: if one
    fails *before it has produced any output* (auth/config/rate-limit errors
    typically fail immediately, before the first token), the next configured
    provider is tried instead automatically. Once a provider has started
    streaming text, a failure is surfaced as-is rather than silently
    retried, since re-sending the prompt to a different provider mid-stream
    would duplicate or corrupt the partial answer already shown.
    """
    if requested_model:
        provider, model = resolve_model(requested_model)
        await _validate_model_override(provider, model)
        candidates = [(provider, model)]
    else:
        # Respect an operator-pinned primary provider (LLM_PROVIDER), but
        # unlike the old strict pick_provider() behaviour, a pinned provider
        # that's down or misconfigured no longer hard-fails the request —
        # it's just tried first, then the rest of _PROVIDER_ORDER acts as
        # its fallback chain. (e.g. LLM_PROVIDER=kimi makes Kimi primary,
        # falling through to DeepSeek/others only if Kimi's call fails.)
        order = list(_PROVIDER_ORDER)
        if LLM_PROVIDER:
            if LLM_PROVIDER not in _PROVIDERS:
                raise LLMError(f"LLM_PROVIDER={LLM_PROVIDER!r} is not a known provider.")
            order = [LLM_PROVIDER] + [p for p in _PROVIDER_ORDER if p != LLM_PROVIDER]
        candidates = [(n, _PROVIDERS[n][1]) for n in order if _PROVIDERS[n][0]]
        if not candidates:
            raise LLMError("No LLM configured. Set an API key for at least one provider.")

    last_error: Optional[Exception] = None
    for i, (provider, model) in enumerate(candidates):
        if not requested_model and provider in AUTO_ROUTE_PROVIDERS and routing_prompt:
            auto = await _auto_pick_model(provider, routing_prompt)
            if auto:
                model = auto
        gen = (_stream_anthropic(msgs, tools_list, model) if provider == "anthropic"
               else _stream_openai(msgs, tools_list, model, provider))
        yielded_any = False
        try:
            async for ev in gen:
                yielded_any = True
                yield ev
            return
        except Exception as e:
            last_error = e
            is_last = i == len(candidates) - 1
            if yielded_any or is_last:
                raise _opaque_llm_error(e, provider)
            log.warning("provider %s failed before producing output (%s: %s); "
                       "trying next configured provider", provider, type(e).__name__, e)
    if last_error:  # pragma: no cover — unreachable, kept as a defensive fallback
        raise _opaque_llm_error(last_error, candidates[-1][0])


# ---------------- TOOL HELPERS ----------------
_OPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod,
        ast.Pow: _op.pow, ast.USub: _op.neg, ast.UAdd: _op.pos}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max,
          "sum": sum, "int": int, "float": float}


def _eval(node: ast.AST) -> Any:
    """Evaluate one node of a whitelisted arithmetic expression."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("numeric literals only")
    if isinstance(node, ast.BinOp):
        fn = _OPS.get(type(node.op))
        if not fn:
            raise ValueError("op not allowed")
        return fn(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = _OPS.get(type(node.op))
        if not fn:
            raise ValueError("unary not allowed")
        return fn(_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("fn not allowed")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e) for e in node.elts)
    raise ValueError(f"not allowed: {type(node).__name__}")


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate an arithmetic expression without exec or eval."""
    return {"expression": expression,
            "result": _eval(ast.parse(expression, mode="eval").body)}


def get_current_time(timezone_name: str = "UTC") -> dict[str, Any]:
    """Return the current time in an IANA timezone."""
    from zoneinfo import ZoneInfo
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    return {"iso": now.isoformat(), "timezone": str(tz), "unix": now.timestamp()}


def _assert_public_url(url: str) -> None:
    """Reject non-http(s) URLs and anything resolving to a non-public address."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("http/https only")
    if not p.hostname:
        raise ValueError("no host")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        raise ValueError("cannot resolve")
    if not infos:
        raise ValueError("cannot resolve")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError("refusing private/internal address")


async def web_fetch(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """Fetch a public URL as text, revalidating the target on every redirect."""
    _assert_public_url(url)
    async with httpx.AsyncClient(timeout=20, follow_redirects=False,
                                 headers={"User-Agent": "YizAI/4.0"}) as c:
        r = await c.get(url)
        for _ in range(WEB_FETCH_MAX_REDIRECTS):
            if r.status_code not in (301, 302, 303, 307, 308):
                break
            location = r.headers.get("location")
            if not location:
                break
            url = str(httpx.URL(str(r.url)).join(location))
            _assert_public_url(url)
            r = await c.get(url)
        else:
            raise ValueError("too many redirects")
    if len(r.content) > WEB_FETCH_MAX_BYTES:
        raise ValueError("response too large")
    text = r.text
    if "<html" in text[:1000].lower():
        from html.parser import HTMLParser

        class _S(HTMLParser):
            """Collects visible text from an HTML document."""
            def __init__(self) -> None:
                """Initialize the text collector."""
                super().__init__()
                self.parts = []
                self._skip = False

            def handle_starttag(self, tag: str, attrs: list) -> None:
                """Start skipping script and style content."""
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag: str) -> None:
                """Stop skipping script and style content."""
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, d: str) -> None:
                """Collect a non-empty text run."""
                if not self._skip and d.strip():
                    self.parts.append(d.strip())

        s = _S()
        s.feed(text)
        text = "\n".join(s.parts)
    return {"url": str(r.url), "status": r.status_code,
            "content": text[:max_chars], "truncated": len(text) > max_chars}


async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web through Tavily."""
    if not TAVILY_API_KEY:
        return {"error": "web_search not configured (set TAVILY_API_KEY)"}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post("https://api.tavily.com/search", json={
            "api_key": TAVILY_API_KEY, "query": query,
            "max_results": max_results, "search_depth": "basic"})
    r.raise_for_status()
    data = r.json()
    return {"query": query, "results": [
        {"title": x.get("title"), "url": x.get("url"),
         "snippet": (x.get("content") or "")[:400]}
        for x in data.get("results", [])]}


def hash_text(text: str, algorithm: str = "sha256") -> dict[str, str]:
    """Hash text with a whitelisted algorithm."""
    if algorithm not in ("md5", "sha1", "sha256", "sha512"):
        raise ValueError("unsupported algorithm")
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return {"algorithm": algorithm, "hash": h.hexdigest()}


async def generate_qr(text: str, size: int = 8) -> dict[str, str]:
    """Render a QR code PNG and store it."""
    import qrcode
    qr = qrcode.QRCode(box_size=size, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    stored = await store_media(buf.getvalue(), "png", "image/png")
    return {"text": text, "image_url": f"/api/media/{stored}"}


# ---------------- MEDIA TOOLS ----------------
async def _generate_image(prompt: str, width: int = 1024, height: int = 1024,
                          style: str = "") -> dict[str, Any]:
    """Generate an image via Stability, falling back to Pollinations."""
    full = f"{prompt}, {style}" if style else prompt
    key = os.getenv("STABILITY_API_KEY", "")
    if key:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    "https://api.stability.ai/v2beta/stable-image/generate/core",
                    headers={"Authorization": f"Bearer {key}", "Accept": "image/*"},
                    files={"none": ""},
                    data={"prompt": full, "output_format": "png",
                          "width": width, "height": height})
            if r.status_code != 200:
                return {"error": f"stability {r.status_code}"}
            stored = await store_media(r.content, "png", "image/png")
            return {"provider": "stability", "prompt": full,
                    "image_url": f"/api/media/{stored}", "width": width, "height": height}
        except httpx.HTTPError as e:
            return {"error": f"stability failed: {e}"}
    url = (f"https://image.pollinations.ai/prompt/"
           f"{quote(full)}?width={width}&height={height}&nologo=true")
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.get(url)
        if r.status_code != 200:
            return {"error": f"pollinations {r.status_code}"}
        stored = await store_media(r.content, "png", "image/png")
        return {"provider": "pollinations", "prompt": full,
                "image_url": f"/api/media/{stored}", "width": width, "height": height}
    except httpx.HTTPError as e:
        return {"error": f"image gen failed: {e}"}


async def _text_to_speech(text: str, lang: str = "en") -> dict[str, Any]:
    """Synthesize short text to an MP3."""
    text = text[:200]
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(
                "https://translate.google.com/translate_tts",
                params={"ie": "UTF-8", "q": text, "tl": lang, "client": "tw-ob"},
                headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return {"error": f"tts {r.status_code}"}
        stored = await store_media(r.content, "mp3", "audio/mpeg")
        return {"text": text, "lang": lang, "audio_url": f"/api/media/{stored}"}
    except httpx.HTTPError as e:
        return {"error": f"tts failed: {e}"}


async def _generate_video(prompt: str, duration: int = 3) -> dict[str, Any]:
    """Generate a short video through Replicate."""
    token = os.getenv("REPLICATE_API_TOKEN", "")
    if not token:
        return {"error": "video generation requires REPLICATE_API_TOKEN"}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json={"version": "3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                  "input": {"prompt": prompt, "video_length": duration}})
    if r.status_code not in (200, 201):
        return {"error": f"replicate {r.status_code}"}
    pred_id = r.json()["id"]
    for _ in range(60):
        await asyncio.sleep(2)
        async with httpx.AsyncClient(timeout=30) as c:
            s = await c.get(f"https://api.replicate.com/v1/predictions/{pred_id}",
                            headers={"Authorization": f"Bearer {token}"})
        data = s.json()
        if data["status"] == "succeeded":
            url = data["output"][0] if isinstance(data["output"], list) else data["output"]
            async with httpx.AsyncClient(timeout=60) as c:
                v = await c.get(url)
            stored = await store_media(v.content, "mp4", "video/mp4")
            return {"provider": "replicate", "prompt": prompt,
                    "video_url": f"/api/media/{stored}"}
        if data["status"] == "failed":
            return {"error": f"video failed: {data.get('error')}"}
    return {"error": "video generation timed out"}


# ---------------- FILE GENERATORS ----------------
def _safe_name(name: Optional[str], default_ext: str) -> str:
    """Sanitize a caller-supplied filename and ensure it has an extension."""
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name or f"file.{default_ext}")[:150]
    if not Path(safe).suffix:
        safe += "." + default_ext
    return safe


async def _generate_text_file(filename: Optional[str], content: str, **_: Any) -> dict[str, Any]:
    """Store arbitrary text as a downloadable file."""
    safe = _safe_name(filename, "txt")
    data = content.encode("utf-8")
    if len(data) > MEDIA_MAX_BYTES:
        return {"error": f"file too large ({len(data)} bytes)"}
    ext = Path(safe).suffix.lstrip(".")
    stored = await store_media(data, ext, mimetypes.guess_type(safe)[0], safe)
    return {"filename": safe, "size": len(data),
            "download_url": f"/api/media/{stored}?name={quote(safe)}"}


async def _generate_csv(filename: Optional[str], headers: Optional[list] = None,
                        rows: Optional[list] = None, **_: Any) -> dict[str, Any]:
    """Build a CSV file from headers and rows."""
    buf = io.StringIO()
    w = _csv.writer(buf)
    if headers:
        w.writerow(headers)
    for r in rows or []:
        w.writerow(r)
    return await _generate_text_file(filename or "data.csv", buf.getvalue())


async def _generate_json_file(filename: Optional[str], data: Any, **_: Any) -> dict[str, Any]:
    """Store a JSON document as a downloadable file."""
    return await _generate_text_file(filename or "data.json",
                                     json.dumps(data, indent=2, default=str))


async def _generate_markdown_file(filename: Optional[str], content: str,
                                  **_: Any) -> dict[str, Any]:
    """Store markdown as a downloadable file."""
    return await _generate_text_file(filename or "document.md", content)


async def _generate_pdf(filename: Optional[str], title: str, sections: Optional[list],
                        **_: Any) -> dict[str, Any]:
    """Render a titled, sectioned PDF."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        return {"error": "reportlab not installed"}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11, leading=16)
    flow = [Paragraph(title or "Document", styles["Title"]), Spacer(1, 20)]
    for sec in sections or []:
        if not isinstance(sec, dict):
            sec = {"body": sec}
        h = sec.get("heading")
        b = sec.get("body", "")
        if h:
            flow.append(Paragraph(_xml_escape(str(h)), styles["Heading2"]))
            flow.append(Spacer(1, 6))
        for para in str(b).split("\n\n"):
            if para.strip():
                safe_para = _xml_escape(para.strip()).replace("\n", "<br/>")
                flow.append(Paragraph(safe_para, body_style))
                flow.append(Spacer(1, 8))
        flow.append(Spacer(1, 12))
    doc.build(flow)
    data = buf.getvalue()
    fname = _safe_name(filename or "document.pdf", "pdf")
    stored = await store_media(data, "pdf", "application/pdf", fname)
    return {"filename": fname, "size": len(data),
            "download_url": f"/api/media/{stored}?name={quote(fname)}"}


async def _generate_xlsx(filename: Optional[str], sheets: Optional[list],
                         **_: Any) -> dict[str, Any]:
    """Render an Excel workbook from sheet definitions."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        return {"error": "openpyxl not installed"}
    wb = Workbook()
    wb.remove(wb.active)
    for s in sheets or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "Sheet")[:31]
        ws = wb.create_sheet(title=name)
        headers = s.get("headers") or []
        if headers:
            for c, h in enumerate(headers, 1):
                cell = ws.cell(row=1, column=c, value=h)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="4B0082")
        for r, row in enumerate(s.get("rows") or [], 2):
            for c, val in enumerate(row, 1):
                ws.cell(row=r, column=c, value=val)
        for col in ws.columns:
            width = max((len(str(c.value or "")) for c in col), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 40)
    if not wb.worksheets:
        wb.create_sheet(title="Sheet1")
    buf = io.BytesIO()
    wb.save(buf)
    data = buf.getvalue()
    fname = _safe_name(filename or "workbook.xlsx", "xlsx")
    stored = await store_media(
        data, "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        fname)
    return {"filename": fname, "size": len(data),
            "download_url": f"/api/media/{stored}?name={quote(fname)}"}


async def _generate_docx(filename: Optional[str], title: str,
                         paragraphs: Optional[list], **_: Any) -> dict[str, Any]:
    """Render a Word document from a title and paragraphs."""
    try:
        from docx import Document
    except ImportError:
        return {"error": "python-docx not installed"}
    doc = Document()
    if title:
        doc.add_heading(title, level=0)
    for p in paragraphs or []:
        if isinstance(p, dict):
            if p.get("heading"):
                doc.add_heading(p["heading"], level=1)
            if p.get("body"):
                doc.add_paragraph(p["body"])
        else:
            doc.add_paragraph(str(p))
    buf = io.BytesIO()
    doc.save(buf)
    data = buf.getvalue()
    fname = _safe_name(filename or "document.docx", "docx")
    stored = await store_media(
        data, "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        fname)
    return {"filename": fname, "size": len(data),
            "download_url": f"/api/media/{stored}?name={quote(fname)}"}


async def _generate_pptx(filename: Optional[str], title: str,
                         slides: Optional[list], **_: Any) -> dict[str, Any]:
    """Render a PowerPoint deck. slides = [{heading, bullets: [str,...]} | {heading, body}]."""
    try:
        from pptx import Presentation
        from pptx.util import Inches
    except ImportError:
        return {"error": "python-pptx not installed"}
    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = title or "Presentation"
    for s in slides or []:
        if not isinstance(s, dict):
            s = {"heading": str(s)}
        sl = prs.slides.add_slide(prs.slide_layouts[1])
        sl.shapes.title.text = str(s.get("heading") or "")
        bullets = s.get("bullets") or ([s["body"]] if s.get("body") else [])
        try:
            body = sl.placeholders[1].text_frame
        except (KeyError, IndexError):
            body = sl.shapes.add_textbox(Inches(0.5), Inches(1.5),
                                         Inches(9), Inches(5)).text_frame
        if bullets:
            body.text = str(bullets[0])
            for b in bullets[1:]:
                body.add_paragraph().text = str(b)
    buf = io.BytesIO()
    prs.save(buf)
    data = buf.getvalue()
    fname = _safe_name(filename or "presentation.pptx", "pptx")
    stored = await store_media(
        data, "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        fname)
    return {"filename": fname, "size": len(data),
            "download_url": f"/api/media/{stored}?name={quote(fname)}"}


def _ocr_image_bytes(data: bytes) -> Optional[str]:
    """Run Tesseract OCR over raw image bytes.

    Returns None (never raises) when pytesseract/Pillow aren't installed,
    the tesseract binary itself is missing from the host, or no text is
    found — any of which should degrade gracefully, not break the caller.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(img).strip() or None
    except pytesseract.pytesseract.TesseractNotFoundError:
        log.warning("pytesseract is installed but the tesseract binary is "
                    "missing from this host (install the tesseract-ocr "
                    "system package)")
        return None
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return None


def _ocr_pdf_pages(path: Path, max_chars: int, max_pages: int = 20) -> Optional[str]:
    """Render a scanned/image-only PDF's pages and OCR each one.

    Uses PyMuPDF (no external binary needed) to rasterize pages, so the only
    system dependency this adds is the tesseract-ocr binary itself.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        doc = fitz.open(str(path))
        chunks, total = [], 0
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = pytesseract.image_to_string(img).strip()
            if text:
                chunks.append(f"[Page {i + 1}]\n{text}")
                total += len(text)
            if total > max_chars:
                break
        doc.close()
        return "\n\n".join(chunks)[:max_chars] or None
    except pytesseract.pytesseract.TesseractNotFoundError:
        log.warning("pytesseract is installed but the tesseract binary is "
                    "missing from this host (install the tesseract-ocr "
                    "system package)")
        return None
    except Exception as e:
        log.warning("PDF OCR failed for %s: %s", path.name, e)
        return None


def _extract_attachment_text(path: Path, content_type: str,
                             max_chars: int = 40000) -> Optional[str]:
    """Best-effort text extraction for PDF/DOCX/PPTX/image uploads.

    Never raises: an unsupported or unparsable file returns None so the
    caller can fall back to the analyze_data/query_data hint instead of
    breaking the chat turn. Image-only ("scanned") PDFs fall through to OCR
    automatically when the embedded-text extraction comes back too sparse.
    """
    ct = (content_type or "").lower()
    suffix = path.suffix.lower()
    try:
        if ct.startswith("image/") or suffix in (
                ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif"):
            return _ocr_image_bytes(path.read_bytes())
        if ct == "application/pdf" or suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages, total = [], 0
            for pg in reader.pages:
                t = pg.extract_text() or ""
                pages.append(t)
                total += len(t)
                if total > max_chars:
                    break
            text = "\n".join(pages).strip()
            # Heuristic: real text-based PDFs average well over 20 chars/page;
            # a lower yield usually means the pages are scanned images.
            if text and len(text) >= 20 * max(1, len(reader.pages)):
                return text[:max_chars]
            ocr_text = _ocr_pdf_pages(path, max_chars)
            return ocr_text or (text[:max_chars] or None)
        if suffix == ".docx" or ct == (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"):
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)[:max_chars] or None
        if suffix == ".pptx" or ct == (
                "application/vnd.openxmlformats-officedocument."
                "presentationml.presentation"):
            from pptx import Presentation
            prs = Presentation(str(path))
            chunks, total = [], 0
            for i, slide in enumerate(prs.slides, 1):
                lines = [f"[Slide {i}]"]
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False) and shape.text_frame.text:
                        lines.append(shape.text_frame.text)
                chunk = "\n".join(lines)
                chunks.append(chunk)
                total += len(chunk)
                if total > max_chars:
                    break
            return "\n\n".join(chunks)[:max_chars] or None
    except ImportError:
        return None
    except Exception as e:
        log.warning("attachment text extraction failed for %s: %s", path.name, e)
        return None
    return None


async def _ocr_image(source: str, user_id: str, **_: Any) -> dict[str, Any]:
    """OCR a stored image or scanned PDF and return the extracted text."""
    path = await _resolve_source(source, user_id)
    if not path:
        return {"error": "no such file, or you don't own it"}
    ctype = mimetypes.guess_type(str(path))[0] or ""
    text = _extract_attachment_text(path, ctype)
    if text is None:
        return {"error": "OCR unavailable (missing pytesseract/tesseract on "
                          "this host, or no text found in the file)"}
    return {"text": text}


async def _transcribe_audio(source: str, user_id: str, **_: Any) -> dict[str, Any]:
    """Transcribe a stored audio file (voice note, recording) via OpenAI."""
    if not OPENAI_API_KEY:
        return {"error": "audio transcription requires OPENAI_API_KEY to be configured"}
    path = await _resolve_source(source, user_id)
    if not path:
        return {"error": "no such file, or you don't own it"}
    try:
        data = path.read_bytes()
    except OSError as e:
        return {"error": f"could not read file: {e}"}
    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{OPENAI_BASE_URL}/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                data={"model": TRANSCRIBE_MODEL},
                files={"file": (path.name, data, ctype)})
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.text[:300]
        except Exception:
            pass
        return {"error": f"transcription failed: HTTP {e.response.status_code}",
                "detail": detail}
    except httpx.HTTPError as e:
        return {"error": f"transcription request failed: {e}"}
    text = (r.json() or {}).get("text", "")
    return {"text": text} if text else {"error": "transcription returned no text"}


# ---------------- DATA ANALYST ----------------
async def _make_chart(chart_type: str, title: str, x: list, y: list,
                      series_label: str = "", x_label: str = "", y_label: str = "",
                      **_: Any) -> dict[str, Any]:
    """Render a line, bar, scatter, or pie chart to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"error": "matplotlib not installed"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ct = (chart_type or "line").lower()
    try:
        if ct == "line":
            ax.plot(x, y, marker="o", label=series_label or "series")
        elif ct == "bar":
            ax.bar(x, y, label=series_label or "series")
        elif ct == "scatter":
            ax.scatter(x, y, label=series_label or "series")
        elif ct == "pie":
            ax.pie(y, labels=x, autopct="%1.1f%%")
            ax.axis("equal")
        else:
            ax.plot(x, y, marker="o")
        if ct != "pie":
            ax.set_xlabel(x_label or "x")
            ax.set_ylabel(y_label or "y")
            if series_label:
                ax.legend()
        ax.set_title(title or "Chart")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    except Exception as e:
        plt.close(fig)
        return {"error": f"chart build failed: {e}"}
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    stored = await store_media(data, "png", "image/png")
    return {"chart_type": ct, "title": title,
            "image_url": f"/api/media/{stored}"}


_DUCK_READERS = {"csv": "read_csv_auto", "parquet": "read_parquet",
                 "pq": "read_parquet", "json": "read_json_auto"}


async def _resolve_source(source: str, user_id: str) -> Optional[Path]:
    """Resolve a caller-supplied stored filename, scoped to its owner.

    Looks the key up in the database first so one user cannot read another
    user's stored file by guessing or copying its stored name.
    """
    if not source or len(source) > 260:
        return None
    try:
        key = media_store._safe_key(source)
    except ValueError:
        return None
    meta = await db.get_media(key, user_id)
    if not meta:
        return None
    return media_store.path(key)


def _duck_load(con: Any, path: Path) -> None:
    """Materialize path into a `data` table, then revoke filesystem access."""
    reader = _DUCK_READERS.get(path.suffix.lstrip(".").lower())
    if not reader:
        raise ValueError(f"unsupported file type: {path.suffix}")
    con.execute(f"CREATE TABLE data AS SELECT * FROM {reader}($path)",
                {"path": str(path)})
    _duck_lockdown(con)


def _duck_lockdown(con: Any) -> None:
    """Disable external access so caller SQL cannot reach the filesystem."""
    for stmt in ("SET disabled_filesystems='LocalFileSystem'",
                 "SET enable_external_access=false",
                 "SET lock_configuration=true"):
        try:
            con.execute(stmt)
        except Exception:
            log.warning("duckdb lockdown statement rejected: %s", stmt)


async def _analyze_data(source: str, user_id: str, **_: Any) -> dict[str, Any]:
    """Profile a stored CSV/Parquet/JSON file, or raw CSV text."""
    try:
        import duckdb
        import pandas as pd  # noqa: F401
    except ImportError:
        return {"error": "duckdb or pandas not installed"}
    path = await _resolve_source(source, user_id) if "." in (source or "") else None
    tmp_dir = None
    try:
        con = duckdb.connect()
        if path:
            _duck_load(con, path)
        else:
            tmp_dir = tempfile.mkdtemp()
            tmp = Path(tmp_dir) / "inline.csv"
            tmp.write_text(source or "", encoding="utf-8")
            _duck_load(con, tmp)
        df = con.execute("SELECT * FROM data").fetchdf()
        numeric = df.select_dtypes(include="number")
        describe = numeric.describe().round(3).to_dict() if not numeric.empty else {}
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "null_counts": {str(c): int(n) for c, n in df.isnull().sum().items()},
            "head": df.head(10).fillna("").to_dict(orient="records"),
            "numeric_summary": describe,
        }
    except Exception as e:
        return {"error": f"analysis failed: {e}"}
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


async def _query_data(source: str, sql: str, user_id: str,
                      **_: Any) -> dict[str, Any]:
    """Run caller SQL over a stored file with DuckDB, filesystem access off."""
    try:
        import duckdb
    except ImportError:
        return {"error": "duckdb not installed"}
    path = await _resolve_source(source, user_id)
    if not path:
        return {"error": "no data source provided"}
    try:
        con = duckdb.connect()
        _duck_load(con, path)
        result = con.execute(sql).fetchdf()
        return {
            "sql": sql,
            "rows": int(result.shape[0]),
            "columns": [str(c) for c in result.columns],
            "data": result.head(DUCKDB_ROW_LIMIT).fillna("").to_dict(orient="records"),
            "truncated": result.shape[0] > DUCKDB_ROW_LIMIT,
        }
    except Exception as e:
        return {"error": f"query failed: {e}"}


# ---------------- TOOL REGISTRY ----------------
class ToolRegistry:
    """Registry of every callable tool exposed to the model."""
    def __init__(self) -> None:
        """Register the built-in tool set."""
        self._tools = {}
        self._register_all()

    def register(self, name: Optional[str], description: str, parameters: dict,
                 fn: Callable, is_async: bool = False, needs_user: bool = False) -> None:
        """Add one tool to the registry."""
        self._tools[name] = {"schema": {"name": name, "description": description,
                                        "parameters": parameters},
                             "fn": fn, "is_async": is_async, "needs_user": needs_user}

    def _register_all(self) -> None:
        """Register every built-in tool."""
        R = self.register
        R("calculator", "Evaluate a numeric expression.",
          {"type": "object", "properties": {"expression": {"type": "string"}},
           "required": ["expression"]},
          lambda expression: calculator(expression))
        R("get_current_time", "Current date/time in an IANA timezone.",
          {"type": "object", "properties": {"timezone_name": {"type": "string"}}},
          lambda timezone_name="UTC": get_current_time(timezone_name))
        R("web_fetch", "Fetch a public URL and return readable text.",
          {"type": "object", "properties": {"url": {"type": "string"}},
           "required": ["url"]},
          lambda url: web_fetch(url), is_async=True)
        R("web_search", "Search the web. Requires TAVILY_API_KEY.",
          {"type": "object", "properties": {"query": {"type": "string"},
                                            "max_results": {"type": "integer"}},
           "required": ["query"]},
          lambda query, max_results=5: web_search(query, max_results), is_async=True)
        R("hash_text", "Hash a string with md5/sha1/sha256/sha512.",
          {"type": "object", "properties": {"text": {"type": "string"},
                                            "algorithm": {"type": "string"}},
           "required": ["text"]},
          lambda text, algorithm="sha256": hash_text(text, algorithm))
        R("generate_qr", "Generate a QR code PNG.",
          {"type": "object", "properties": {"text": {"type": "string"},
                                            "size": {"type": "integer"}},
           "required": ["text"]},
          generate_qr, is_async=True)
        R("run_code",
          "Execute code in a sandbox. Languages: python, javascript, typescript, "
          "go, rust, java, c, cpp, csharp, bash, ruby, php, sql, kotlin, swift, lua.",
          {"type": "object",
           "properties": {"language": {"type": "string"},
                          "code": {"type": "string"},
                          "stdin": {"type": "string"}},
           "required": ["language", "code"]},
          self._run_code, is_async=True, needs_user=True)
        R("save_note", "Persist a note for later recall.",
          {"type": "object", "properties": {"content": {"type": "string"}},
           "required": ["content"]},
          self._save_note, is_async=True, needs_user=True)
        R("list_notes", "List saved notes.",
          {"type": "object", "properties": {}},
          self._list_notes, is_async=True, needs_user=True)
        R("n8n_trigger", "Trigger an n8n workflow via webhook.",
          {"type": "object",
           "properties": {"workflow": {"type": "string"},
                          "payload": {"type": "object"}},
           "required": ["workflow"]},
          self._n8n_trigger, is_async=True)
        R("mcp_list_servers", "List configured MCP servers.",
          {"type": "object", "properties": {}},
          self._mcp_list, is_async=True)
        R("mcp_list_tools", "List tools from an MCP server.",
          {"type": "object", "properties": {"server": {"type": "string"}},
           "required": ["server"]},
          self._mcp_list_tools, is_async=True)
        R("mcp_call", "Call a tool on an MCP server.",
          {"type": "object",
           "properties": {"server": {"type": "string"},
                          "tool": {"type": "string"},
                          "args": {"type": "object"}},
           "required": ["server", "tool"]},
          self._mcp_call, is_async=True)
        R("generate_image", "Generate an image from a text prompt.",
          {"type": "object",
           "properties": {"prompt": {"type": "string"},
                          "width": {"type": "integer"},
                          "height": {"type": "integer"},
                          "style": {"type": "string"}},
           "required": ["prompt"]},
          _generate_image, is_async=True)
        R("text_to_speech", "Convert text to speech. Returns MP3.",
          {"type": "object",
           "properties": {"text": {"type": "string"},
                          "lang": {"type": "string"}},
           "required": ["text"]},
          _text_to_speech, is_async=True)
        R("generate_video",
          "Generate a short video (2-5s). Requires REPLICATE_API_TOKEN.",
          {"type": "object",
           "properties": {"prompt": {"type": "string"},
                          "duration": {"type": "integer"}},
           "required": ["prompt"]},
          _generate_video, is_async=True)
        R("generate_text_file",
          "Save any text file (code, script, HTML, markdown, plain text). "
          "Returns a download URL.",
          {"type": "object",
           "properties": {"filename": {"type": "string"},
                          "content": {"type": "string"}},
           "required": ["filename", "content"]},
          _generate_text_file, is_async=True)
        R("generate_csv", "Build a CSV file from headers and rows.",
          {"type": "object",
           "properties": {"filename": {"type": "string"},
                          "headers": {"type": "array"},
                          "rows": {"type": "array"}},
           "required": ["rows"]},
          _generate_csv, is_async=True)
        R("generate_json_file", "Save arbitrary JSON to a .json file.",
          {"type": "object",
           "properties": {"filename": {"type": "string"}, "data": {}},
           "required": ["data"]},
          _generate_json_file, is_async=True)
        R("generate_markdown_file", "Save markdown to a .md file.",
          {"type": "object",
           "properties": {"filename": {"type": "string"},
                          "content": {"type": "string"}},
           "required": ["content"]},
          _generate_markdown_file, is_async=True)
        R("generate_pdf",
          "Generate a formatted PDF from a title and sections [{heading, body}].",
          {"type": "object",
           "properties": {"filename": {"type": "string"},
                          "title": {"type": "string"},
                          "sections": {"type": "array"}},
           "required": ["title", "sections"]},
          _generate_pdf, is_async=True)
        R("generate_xlsx",
          "Generate an Excel workbook. sheets = [{name, headers, rows}].",
          {"type": "object",
           "properties": {"filename": {"type": "string"},
                          "sheets": {"type": "array"}},
           "required": ["sheets"]},
          _generate_xlsx, is_async=True)
        R("generate_docx",
          "Generate a Word document from a title and paragraphs.",
          {"type": "object",
           "properties": {"filename": {"type": "string"},
                          "title": {"type": "string"},
                          "paragraphs": {"type": "array"}},
           "required": ["title", "paragraphs"]},
          _generate_docx, is_async=True)
        R("generate_pptx",
          "Generate a PowerPoint deck. slides = [{heading, bullets: [str,...]}].",
          {"type": "object",
           "properties": {"filename": {"type": "string"},
                          "title": {"type": "string"},
                          "slides": {"type": "array"}},
           "required": ["title", "slides"]},
          _generate_pptx, is_async=True)
        R("make_chart",
          "Generate a chart image (line, bar, scatter, pie) from x/y data.",
          {"type": "object",
           "properties": {"chart_type": {"type": "string"},
                          "title": {"type": "string"},
                          "x": {"type": "array"},
                          "y": {"type": "array"},
                          "series_label": {"type": "string"},
                          "x_label": {"type": "string"},
                          "y_label": {"type": "string"}},
           "required": ["chart_type", "x", "y"]},
          _make_chart, is_async=True)
        R("ocr_image",
          "Extract text from a stored image or scanned PDF using OCR.",
          {"type": "object",
           "properties": {"source": {"type": "string"}},
           "required": ["source"]},
          _ocr_image, is_async=True, needs_user=True)
        R("transcribe_audio",
          "Transcribe a stored audio file (voice note, recording) to text. "
          "Requires OPENAI_API_KEY.",
          {"type": "object",
           "properties": {"source": {"type": "string"}},
           "required": ["source"]},
          _transcribe_audio, is_async=True, needs_user=True)
        R("analyze_data",
          "Analyze a CSV/Parquet/JSON file (pass the stored filename) or raw CSV text.",
          {"type": "object",
           "properties": {"source": {"type": "string"}},
           "required": ["source"]},
          _analyze_data, is_async=True, needs_user=True)
        R("query_data",
          "Run SQL over an uploaded CSV/Parquet/JSON file using DuckDB.",
          {"type": "object",
           "properties": {"source": {"type": "string"},
                          "sql": {"type": "string"}},
           "required": ["source", "sql"]},
          _query_data, is_async=True, needs_user=True)

    async def _run_code(self, user_id: str, language: str, code: str,
                        stdin: Optional[str] = "") -> dict[str, Any]:
        """Run code in the sandbox and record the attempt."""
        try:
            result = await sandbox.execute(language, code, stdin, user_id=user_id)
        except SandboxError as e:
            return {"error": str(e)}
        if "error" in result:
            return result
        try:
            await db.record_code_run(user_id, result.get("provider", "unknown"),
                                     result.get("language", language), code,
                                     result.get("stdout", ""), result.get("stderr", ""),
                                     result.get("exit_code"))
        except Exception:
            pass
        return result

    async def _save_note(self, user_id: str, content: str) -> dict[str, Any]:
        """Persist a note for the calling user."""
        return {"saved": True, "id": await db.add_note(user_id, content)}

    async def _list_notes(self, user_id: str) -> dict[str, Any]:
        """List the calling user's notes."""
        return {"notes": await db.list_notes(user_id)}

    async def _n8n_trigger(self, workflow: str, payload: Optional[dict] = None) -> dict[str, Any]:
        """Trigger an n8n workflow."""
        return await n8n_client.trigger(workflow, payload or {})

    async def _mcp_list(self) -> dict[str, Any]:
        """List configured MCP servers."""
        return {"servers": mcp_client.list_servers()}

    async def _mcp_list_tools(self, server: str) -> dict[str, Any]:
        """List one MCP server's tools."""
        return {"server": server, "tools": await mcp_client.list_tools(server)}

    async def _mcp_call(self, server: str, tool: str,
                        args: Optional[dict] = None) -> dict[str, Any]:
        """Call a tool on an MCP server."""
        return await mcp_client.call_tool(server, tool, args or {})

    def schemas(self) -> list[dict]:
        """Every registered tool schema."""
        return [t["schema"] for t in self._tools.values()]

    def schema_names(self) -> list[str]:
        """Names of every registered tool, for plugin name-collision checks."""
        return list(self._tools.keys())

    @staticmethod
    def _clean_args(schema: dict, args: Any) -> dict:
        """Drop hallucinated keys so an odd tool call cannot raise TypeError."""
        if not isinstance(args, dict):
            return {}
        allowed = (schema.get("parameters") or {}).get("properties")
        if not isinstance(allowed, dict) or not allowed:
            return args
        return {k: v for k, v in args.items() if k in allowed}

    async def call(self, name: str, args: Any, user_id: str) -> str:
        """Invoke a registered tool and return its JSON-encoded result."""
        if name not in self._tools:
            return json.dumps({"error": f"unknown tool: {name}"})
        t = self._tools[name]
        clean = self._clean_args(t["schema"], args)
        try:
            if t["needs_user"]:
                fn = t["fn"](user_id=user_id, **clean)
            else:
                fn = t["fn"](**clean)
            result = await fn if t["is_async"] else fn
            return json.dumps(result, default=str)
        except TypeError as e:
            log.warning("tool %s rejected arguments: %s", name, e)
            return json.dumps({"error": f"invalid arguments for {name}"})
        except Exception as e:
            log.warning("tool %s failed: %s", name, e)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})


tools = ToolRegistry()


# ---------------- FRONTEND ----------------
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Yiz AI</title>
<style>
:root{--bg:#07070f;--panel:#0d0d1a;--line:rgba(120,110,255,.18);--txt:#e8e9f3;--muted:#8b8ba7;--acc:#7c6cff;--acc2:#00d4a8;--mono:ui-monospace,Menlo,monospace}
*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%}
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--txt);overflow:hidden;display:flex;background-image:radial-gradient(900px 500px at 12% -10%,rgba(124,108,255,.13),transparent 60%),radial-gradient(800px 500px at 95% 110%,rgba(0,212,168,.09),transparent 60%)}
aside{width:260px;flex:0 0 260px;background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;transition:margin-left .25s}
aside.hide{margin-left:-260px}
.brand{padding:18px 16px 12px;display:flex;gap:10px;align-items:center}
.logo{width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:grid;place-items:center;font-weight:800;color:#0b0b16;font-size:14px}
.brand h1{font-size:15px;font-weight:650}.brand small{color:var(--muted);font-size:11px;display:block;font-weight:400}
.nc{margin:6px 12px 10px;padding:10px 14px;border-radius:11px;cursor:pointer;background:linear-gradient(135deg,rgba(124,108,255,.22),rgba(0,212,168,.14));border:1px solid var(--line);color:var(--txt);font-weight:600;font-size:13.5px;display:flex;gap:8px;align-items:center}
.nc:hover{border-color:rgba(124,108,255,.5)}
.convs{flex:1;overflow-y:auto;padding:4px 8px 10px}
.convs h2{font-size:10px;letter-spacing:.9px;text-transform:uppercase;color:var(--muted);padding:10px 8px 6px}
.conv{padding:8px 10px;border-radius:9px;cursor:pointer;font-size:13px;color:#c9c9dd;display:flex;justify-content:space-between;gap:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.conv:hover{background:rgba(255,255,255,.045)}.conv.active{background:rgba(124,108,255,.16);color:#fff}
.conv .d{opacity:0;color:var(--muted);font-size:14px}.conv:hover .d{opacity:1}.conv .d:hover{color:#ff6b81}
.sf{padding:10px 12px;border-top:1px solid var(--line);display:flex;gap:8px;align-items:center;font-size:11.5px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--acc2);box-shadow:0 0 8px var(--acc2);flex:0 0 7px}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{height:54px;display:flex;align-items:center;gap:10px;padding:0 16px;border-bottom:1px solid var(--line);background:rgba(7,7,15,.72);backdrop-filter:blur(12px)}
.ib{background:none;border:1px solid var(--line);color:var(--muted);width:32px;height:32px;border-radius:8px;cursor:pointer;font-size:14px;display:grid;place-items:center}
.ib:hover{color:var(--txt);border-color:rgba(124,108,255,.5)}
.title{font-weight:600;font-size:14px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mp{font-size:11px;color:var(--muted);border:1px solid var(--line);padding:3px 9px;border-radius:999px;font-family:var(--mono)}
.scroll{flex:1;overflow-y:auto}
.thr{max-width:800px;margin:0 auto;padding:26px 22px 40px;display:flex;flex-direction:column;gap:18px}
.wel{text-align:center;padding:50px 20px}
.big{width:60px;height:60px;border-radius:17px;margin:0 auto 18px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:grid;place-items:center;font-size:26px;font-weight:800;color:#0b0b16;box-shadow:0 0 40px rgba(124,108,255,.35)}
.wel h2{font-size:24px;font-weight:700;margin-bottom:6px}
.wel p{color:var(--muted);font-size:14px;max-width:460px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:26px;max-width:680px;margin-inline:auto}
.cd{text-align:left;padding:13px 14px;border-radius:12px;cursor:pointer;background:var(--panel);border:1px solid var(--line)}
.cd:hover{border-color:rgba(124,108,255,.5);transform:translateY(-2px)}
.cd b{display:block;font-size:13px;margin-bottom:2px}.cd span{color:var(--muted);font-size:12px}
.msg{display:flex;gap:12px}
.av{width:28px;height:28px;border-radius:8px;flex:0 0 28px;display:grid;place-items:center;font-size:11px;font-weight:700}
.msg.user .av{background:#2a2450;color:#b9aaff}
.msg.assistant .av{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#0b0b16}
.bub{flex:1;min-width:0}.who{font-size:11.5px;font-weight:600;color:var(--muted);margin-bottom:5px}
.body{font-size:14.5px;overflow-wrap:anywhere}
.body p{margin:0 0 10px}.body p:last-child{margin-bottom:0}
.body ul,.body ol{margin:0 0 10px 20px}.body li{margin-bottom:3px}
.body code{font-family:var(--mono);font-size:12.5px;background:rgba(255,255,255,.07);padding:2px 5px;border-radius:5px}
.body pre{background:#08080f;border:1px solid var(--line);border-radius:11px;padding:12px 14px;overflow-x:auto;margin:0 0 10px}
.body pre code{background:none;padding:0;font-size:12.3px}.body a{color:var(--acc2)}
.body img{max-width:100%;border-radius:12px;margin-top:10px;display:block}
.body audio{width:100%;margin-top:10px}
.body video{max-width:100%;border-radius:12px;margin-top:10px}
.body .filelink{display:inline-flex;align-items:center;gap:6px;margin-top:8px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:rgba(124,108,255,.08);color:var(--txt);text-decoration:none;font-size:13px}
.body .filelink:hover{border-color:rgba(124,108,255,.5)}
.cur{display:inline-block;width:7px;height:16px;vertical-align:-3px;background:var(--acc);animation:bl 1s steps(2) infinite;border-radius:1px}
@keyframes bl{0%,50%{opacity:1}50.01%,100%{opacity:0}}
.tl{margin:2px 0 10px;border:1px solid var(--line);border-radius:11px;background:rgba(124,108,255,.05);font-size:12.5px;overflow:hidden}
.th{padding:8px 12px;display:flex;gap:8px;align-items:center;cursor:pointer;font-family:var(--mono);font-size:12px;color:#b9aaff}
.sp{width:10px;height:10px;border:2px solid rgba(185,170,255,.3);border-top-color:#b9aaff;border-radius:50%;animation:sp .7s linear infinite;flex:0 0 10px}
@keyframes sp{to{transform:rotate(360deg)}}
.tl.done .sp{animation:none;border:none;background:var(--acc2);box-shadow:0 0 7px var(--acc2)}
.tb{display:none;padding:8px 12px;font-family:var(--mono);font-size:11.5px;color:var(--muted);white-space:pre-wrap;max-height:260px;overflow:auto;border-top:1px solid var(--line)}
.tl.open .tb{display:block}
.cw{border-top:1px solid var(--line);background:rgba(7,7,15,.82);backdrop-filter:blur(12px);padding:12px 20px 16px}
.cp{max-width:800px;margin:0 auto;position:relative}
.cp textarea{width:100%;resize:none;background:var(--panel);color:var(--txt);border:1px solid var(--line);border-radius:15px;padding:13px 96px 13px 15px;font:inherit;font-size:14.5px;max-height:180px;outline:none;display:block}
.cp textarea:focus{border-color:rgba(124,108,255,.55);box-shadow:0 0 0 3px rgba(124,108,255,.11)}
.cp button{position:absolute;bottom:8px;border-radius:10px;border:none;cursor:pointer;font-size:15px;display:grid;place-items:center}
#clip{right:50px;width:34px;height:34px;background:var(--panel);color:var(--muted);border:1px solid var(--line)}
#clip:hover{color:var(--txt);border-color:rgba(124,108,255,.5)}
#send{right:8px;width:34px;height:34px;background:linear-gradient(135deg,var(--acc),var(--acc2));color:#0b0b16}
#send:disabled{opacity:.32;cursor:not-allowed}
.hint{max-width:800px;margin:8px auto 0;font-size:11px;color:var(--muted);display:flex;justify-content:space-between}
#attBar{display:none;gap:6px;flex-wrap:wrap;margin-bottom:8px}
#attBar .chip{display:flex;align-items:center;gap:6px;padding:5px 10px;background:rgba(124,108,255,.14);border:1px solid var(--line);border-radius:20px;font-size:12px;color:#c9c9dd}
#attBar .chip button{background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px;padding:0 2px}
#attBar .chip button:hover{color:#ff6b81}
.ov{position:fixed;inset:0;background:rgba(3,3,8,.78);backdrop-filter:blur(4px);display:none;place-items:center;z-index:50;padding:20px}
.ov.show{display:grid}
.md{background:var(--panel);border:1px solid var(--line);border-radius:16px;width:100%;max-width:420px;padding:22px}
.md h3{font-size:16px;margin-bottom:5px}.md p.sub{color:var(--muted);font-size:12.5px;margin-bottom:16px}
.fl{margin-bottom:12px}.fl label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}
.fl input{width:100%;padding:10px 12px;border-radius:9px;font:inherit;font-size:13.5px;background:var(--bg);color:var(--txt);border:1px solid var(--line);outline:none}
.fl input:focus{border-color:rgba(124,108,255,.55)}
.mact{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
.bt{padding:9px 16px;border-radius:9px;border:1px solid var(--line);background:none;color:var(--txt);cursor:pointer;font-size:13px;font-weight:550}
.bt.prim{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#0b0b16;border:none}
.bt:hover{opacity:.88}
.oauth{display:flex;flex-direction:column;gap:8px;margin-top:12px}
.oauth a{display:block;text-align:center;padding:10px;border-radius:9px;border:1px solid var(--line);color:var(--txt);text-decoration:none;font-size:13px;font-weight:550}
.oauth a:hover{border-color:rgba(124,108,255,.5)}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(80px);background:#1a1a2e;border:1px solid var(--line);padding:10px 18px;border-radius:11px;font-size:13px;z-index:60;transition:transform .28s cubic-bezier(.34,1.4,.64,1);max-width:90vw}
.toast.show{transform:translateX(-50%) translateY(0)}.toast.err{border-color:rgba(255,107,129,.5);color:#ffb3bd}
.md.wide{max-width:560px;max-height:80vh;overflow-y:auto}
.mcpsrv{border:1px solid var(--line);border-radius:11px;padding:12px 14px;margin-bottom:10px}
.mcpsrv h4{margin:0 0 4px;font-size:13.5px;display:flex;align-items:center;gap:8px}
.mcpsrv .badge{font-size:10.5px;padding:2px 7px;border-radius:20px;font-weight:600}
.mcpsrv .badge.ok{background:rgba(66,214,146,.15);color:#42d692}
.mcpsrv .badge.bad{background:rgba(255,107,129,.15);color:#ff6b81}
.mcpsrv .url{font-size:11.5px;color:var(--muted);word-break:break-all;margin:0 0 8px}
.mcptool{font-size:12px;color:#c9c9dd;padding:3px 0;border-top:1px solid rgba(255,255,255,.05)}
.mcptool b{color:var(--txt);font-weight:600}
.mcpempty{color:var(--muted);font-size:13px;text-align:center;padding:30px 10px}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:rgba(124,108,255,.2);border-radius:4px}
@media(max-width:820px){aside{position:fixed;z-index:40;height:100%;box-shadow:24px 0 60px rgba(0,0,0,.6)}.thr,.cp,.hint{padding-inline:14px}.cards{grid-template-columns:1fr}}
</style></head><body>
<aside id="sb"><div class="brand"><div class="logo">Y</div><div><h1>Yiz AI</h1><small>assistant</small></div></div>
<button class="nc" id="nc"><span>+</span> New chat</button>
<div class="convs" id="cl"><h2>Recent</h2></div>
<div class="sf"><span class="dot" id="sd"></span><span id="st">connecting…</span>
<button class="ib" id="toolsBtn" title="Tools & MCP" style="width:26px;height:26px;font-size:13px;margin-left:auto">🔌</button>
<button class="ib" id="lo" title="Sign out" style="width:26px;height:26px;font-size:12px">⏻</button></div></aside>
<main><header><button class="ib" id="ts">☰</button><div class="title" id="ct">New chat</div><select class="mp" id="ms" title="Model" style="background:var(--panel);color:var(--muted);border:1px solid var(--line);border-radius:8px;padding:4px 8px;max-width:230px;font-size:12px"><option value="">auto</option></select></header>
<div class="scroll" id="sc"><div class="thr" id="th"></div></div>
<div class="cw"><div class="cp">
<div id="attBar"></div>
<textarea id="ip" rows="1" placeholder="Ask anything…  drop a file or click 📎"></textarea>
<button id="clip" title="Attach file">📎</button>
<button id="send" title="Send">↑</button>
<input type="file" id="fileInput" style="display:none" multiple>
</div>
<div class="hint"><span id="ht">Tools available</span><span id="hn"></span></div></div></main>
<div class="ov" id="to"><div class="md wide">
<h3 style="margin:0 0 4px">Tools &amp; MCP</h3><p class="sub" style="margin:0 0 16px">Built-in tools plus any connected Model Context Protocol servers.</p>
<div id="toBody"><p class="mcpempty">Loading…</p></div>
<div class="mact"><button class="bt" id="toClose" type="button">Close</button></div>
</div></div>

<div class="ov" id="ao"><div class="md">
<h3 id="at">Sign in to Yiz AI</h3><p class="sub" id="as">Your chats are private to your account.</p>
<div class="fl"><label>Email</label><input id="ae" type="email" autocomplete="email" placeholder="you@example.com"></div>
<div class="fl"><label>Password</label><input id="ap" type="password" autocomplete="current-password" placeholder="at least 8 characters"></div>
<div class="fl" id="anf" style="display:none"><label>Name (optional)</label><input id="an" type="text" placeholder="Your name"></div>
<div id="aerr" style="color:#ffb3bd;font-size:12.5px;min-height:16px;margin-top:2px"></div>
<div class="mact" style="justify-content:space-between"><button class="bt" id="atog" type="button">Create account</button><button class="bt prim" id="asub" type="button">Sign in</button></div>
<div class="oauth" id="oap"></div></div></div>
<div class="toast" id="tst"></div>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.11/dist/purify.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/common.min.js"></script>
<script>
(function(){
const $=s=>document.querySelector(s);
const API=localStorage.getItem('yiz_base')||'';
const H={'Content-Type':'application/json','X-Requested-With':'yiz'};
let cid=null,streaming=false;
marked.setOptions({breaks:true,gfm:true});
const rmd=t=>DOMPurify.sanitize(marked.parse(t||''));
function hl(root){if(!root||!window.hljs)return;root.querySelectorAll('pre code:not([data-hl])').forEach(el=>{try{hljs.highlightElement(el);el.setAttribute('data-hl','1')}catch{}})}
const esc=s=>String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
function toast(m,e){const t=$('#tst');t.textContent=m;t.className='toast show'+(e?' err':'');clearTimeout(toast._);toast._=setTimeout(()=>t.className='toast',3400)}
async function jf(path,opts){const r=await fetch(API+path,{credentials:'include',headers:H,...opts});if(r.status===401){authShow();throw new Error('sign in required')}return r}
async function health(){try{const d=await(await fetch(API+'/api/health',{credentials:'include'})).json();
if(d.ok&&d.provider){$('#sd').style.background='var(--acc2)';$('#st').textContent='online';
$('#ht').textContent=d.provider+' · '+(d.model||'unknown')+(d.sandbox?' · '+d.sandbox:'')+((d.providers||[]).length>1?' · '+d.providers.length+' providers':'')+((d.mcp_servers||[]).length?' · '+d.mcp_servers.length+' MCP':'');
const s=$('#ms');if(s&&s.options.length)s.options[0].textContent='auto ('+d.provider+':'+(d.model||'?')+')'}
else{$('#sd').style.background='#ffb020';$('#st').textContent='no model';$('#ht').textContent=d.warning||d.error||''}}
catch{$('#sd').style.background='#ff6b81';$('#st').textContent='offline'}}
async function loadModels(){const s=$('#ms');if(!s)return;
try{
const r=await jf('/api/models');
if(!r.ok)throw new Error('model listing failed: HTTP '+r.status);
const d=await r.json();
const saved=localStorage.getItem('yiz_model')||'';
s.innerHTML='<option value="">auto'+(d.default?' ('+d.default+')':'')+'</option>';
for(const [prov,list] of Object.entries(d.providers||{})){
const g=document.createElement('optgroup');g.label=prov;
for(const m of list){const o=document.createElement('option');o.value=prov+':'+m;o.textContent=m;g.appendChild(o)}
s.appendChild(g)}
if(saved&&[...s.options].some(o=>o.value===saved))s.value=saved;
s.onchange=()=>localStorage.setItem('yiz_model',s.value)
}catch(e){console.warn('Model listing failed:',e);s.innerHTML='<option value="">auto</option>'}}
async function loadMcpStatus(){const body=$('#toBody');body.innerHTML='<p class="mcpempty">Loading…</p>';
try{
const r=await jf('/api/mcp/status');
if(!r.ok)throw new Error('HTTP '+r.status);
const d=await r.json();
let html='<p style="font-size:12.5px;color:var(--muted);margin:0 0 14px">'+d.builtin_tool_count+' built-in tools always available.</p>';
if(!d.servers.length&&!d.rejected.length){
html+='<p class="mcpempty">No MCP servers configured.<br>Set <code>MCP_SERVERS=name=https://host</code> in your environment to connect one.</p>'}
else{
for(const s of d.servers){
html+='<div class="mcpsrv"><h4>'+esc(s.name)+' <span class="badge '+(s.status==='ok'?'ok':'bad')+'">'+(s.status==='ok'?s.tools.length+' tools':'unreachable')+'</span></h4>'+
'<p class="url">'+esc(s.url)+'</p>'+
s.tools.map(t=>'<div class="mcptool"><b>'+esc(t.name)+'</b>'+(t.description?' — '+esc(t.description):'')+'</div>').join('')+
'</div>'}
for(const s of d.rejected){
html+='<div class="mcpsrv"><h4>'+esc(s.name)+' <span class="badge bad">rejected</span></h4><p class="url">'+esc(s.url)+'</p><p style="font-size:12px;color:#ff6b81;margin:0">'+esc(s.reason)+'</p></div>'}}
body.innerHTML=html
}catch(e){body.innerHTML='<p class="mcpempty">Could not load MCP status: '+esc(e.message)+'</p>'}}
const mk=(t,c,h)=>{const n=document.createElement(t);if(c)n.className=c;if(h!=null)n.innerHTML=h;return n};
async function loadConvs(){const L=$('#cl');try{const d=await(await jf('/api/conversations')).json();
L.querySelectorAll('.conv,h2:not(:first-child)').forEach(n=>n.remove());
if(!d.conversations.length){L.appendChild(mk('h2',null,'No chats yet'));return}
for(const c of d.conversations){const n=mk('div','conv');n.dataset.id=c.id;
n.innerHTML='<span>'+esc(c.title)+'</span><span class="d">×</span>';
n.querySelector('span:first-child').onclick=()=>openConv(c.id,c.title);
n.querySelector('.d').onclick=async e=>{e.stopPropagation();try{await jf('/api/conversations/'+c.id,{method:'DELETE'});if(cid===c.id)newChat();loadConvs()}catch(err){toast(err.message,1)}};
L.appendChild(n)}}catch{L.appendChild(mk('h2',null,'Backend offline'))}}
const markA=()=>document.querySelectorAll('.conv').forEach(n=>n.classList.toggle('active',n.dataset.id===cid));
function clearTh(){$('#th').innerHTML=''}
function welcome(){clearTh();const w=mk('div','wel');
w.innerHTML='<div class="big">Y</div><h2>How can I help?</h2><p>I run code, generate images and files, analyze data, search the web, and trigger automations.</p><div class="cards">'+
'<div class="cd" data-p="Generate an image of a cyberpunk city at night, neon and rain."><b>Generate an image</b><span>from a prompt</span></div>'+
'<div class="cd" data-p="Write a python script that fetches JSON from an API and saves it as CSV. Then save the script as fetch_csv.py."><b>Write & save code</b><span>any language</span></div>'+
'<div class="cd" data-p="Create a PDF report titled Q4 Sales with three sections: Overview, Numbers, Recommendations."><b>Build a PDF</b><span>download instantly</span></div>'+
'<div class="cd" data-p="Search the web for the latest Python release and summarize what changed."><b>Search the web</b><span>current information</span></div>'+
'<div class="cd" data-p="Here is CSV data:\\nname,revenue\\nA,100\\nB,250\\nC,180\\nD,320\\nAnalyze it and make a bar chart."><b>Analyze data</b><span>profile, SQL, chart</span></div></div>';
w.querySelectorAll('.cd').forEach(c=>c.onclick=()=>{$('#ip').value=c.dataset.p.replace(/\\n/g,String.fromCharCode(10));grow();send()});
$('#th').appendChild(w);$('#ct').textContent='New chat'}
function addMsg(role,text=''){const w=mk('div','msg '+role);
w.appendChild(mk('div','av',role==='user'?'You':'Y'));const b=mk('div','bub');
b.appendChild(mk('div','who',role==='user'?'You':'Yiz AI'));const bd=mk('div','body');
bd.innerHTML=role==='user'?esc(text).replace(/\n/g,'<br>'):rmd(text);
if(role!=='user')hl(bd);
b.appendChild(bd);w.appendChild(b);$('#th').appendChild(w);return{w,bd}}
function addTool(name,args){const b=mk('div','tl open');
b.innerHTML='<div class="th"><span class="sp"></span><span>'+esc(name)+'</span><span style="opacity:.55;font-size:11px">running…</span></div><div class="tb">'+esc(JSON.stringify(args,null,2))+'</div>';
b.querySelector('.th').onclick=()=>b.classList.toggle('open');
const last=$('#th').lastElementChild;
if(last&&last.classList.contains('msg'))last.querySelector('.bub').appendChild(b);else $('#th').appendChild(b);
return b}
function finishTool(b,result){b.classList.add('done');b.querySelector('.th span:last-child').textContent='done';
b.querySelector('.tb').textContent=JSON.stringify(b._args||{},null,2)+'\n\n─── result ───\n'+result}
const dwn=()=>{const s=$('#sc');s.scrollTop=s.scrollHeight};
function appendMedia(b,pl){
  if(pl.type==='image'){const img=document.createElement('img');img.src=API+pl.url;img.loading='lazy';img.alt=pl.name||'generated image';b.appendChild(img)}
  else if(pl.type==='audio'){const au=document.createElement('audio');au.controls=true;au.src=API+pl.url;b.appendChild(au)}
  else if(pl.type==='video'){const v=document.createElement('video');v.controls=true;v.playsInline=true;v.src=API+pl.url;b.appendChild(v)}
  else if(pl.type==='file'){const a=document.createElement('a');a.className='filelink';a.href=API+pl.url;a.download=pl.name||'';a.rel='noopener';a.textContent='\u2193  '+(pl.name||'Download file');b.appendChild(a)}
  dwn()
}
const uploadedFiles=[];
function renderAtts(){const bar=$('#attBar');bar.innerHTML='';
  if(!uploadedFiles.length){bar.style.display='none';return}
  bar.style.display='flex';
  uploadedFiles.forEach((f,i)=>{const chip=document.createElement('div');chip.className='chip';
    chip.innerHTML='📄 '+esc(f.filename)+' ('+Math.round(f.size/1024)+' KB) <button data-i="'+i+'">×</button>';
    chip.querySelector('button').onclick=()=>{uploadedFiles.splice(i,1);renderAtts()};
    bar.appendChild(chip)})}
async function uploadOne(file){
  const fd=new FormData();fd.append('file',file);
  try{const r=await fetch(API+'/api/upload',{method:'POST',credentials:'include',headers:{'X-Requested-With':'yiz'},body:fd});
    if(!r.ok){toast('upload failed: '+r.status,1);return null}
    const d=await r.json();uploadedFiles.push(d);renderAtts();return d
  }catch(err){toast('upload error: '+err.message,1);return null}}
$('#clip').onclick=()=>$('#fileInput').click();
$('#fileInput').onchange=async e=>{for(const f of e.target.files)await uploadOne(f);e.target.value=''};
const dropZone=$('#ip').closest('.cp');
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.style.outline='2px dashed var(--acc)'});
dropZone.addEventListener('dragleave',()=>dropZone.style.outline='');
dropZone.addEventListener('drop',async e=>{e.preventDefault();dropZone.style.outline='';
  for(const f of e.dataTransfer.files)await uploadOne(f)});
async function send(){if(streaming)return;const ip=$('#ip');const p=ip.value.trim();if(!p)return;
const w=document.querySelector('.wel');if(w)w.remove();
addMsg('user',p);ip.value='';grow();dwn();
streaming=true;$('#send').disabled=true;
const{bd:b}=addMsg('assistant','');b.innerHTML='';
const txtEl=mk('div');const medEl=mk('div');b.appendChild(txtEl);b.appendChild(medEl);
txtEl.innerHTML='<span class="cur"></span>';
let txt='';let streamError=false;let mediaCount=0;const tbs=new Map();
// Text is re-rendered on every token; media nodes live in their own container so
// each element is created exactly once and never duplicated or reloaded.
const render=cur=>{txtEl.innerHTML=rmd(txt)+(cur?'<span class="cur"></span>':'')};
try{
const endpoint=uploadedFiles.length?'/api/chat-with-files':'/api/chat';
const payload={prompt:p,conversation_id:cid};
const _m=($('#ms')&&$('#ms').value)||'';if(_m)payload.model=_m;
if(uploadedFiles.length)payload.attachment_ids=uploadedFiles.map(f=>f.id);
const r=await fetch(API+endpoint,{method:'POST',credentials:'include',headers:H,body:JSON.stringify(payload)});
if(r.status===401){authShow();throw new Error('sign in required')}
if(!r.ok||!r.body)throw new Error('HTTP '+r.status+' — '+(await r.text()).slice(0,200));
uploadedFiles.length=0;renderAtts();
const rd=r.body.getReader();const dc=new TextDecoder();let buf='';
while(true){const{value,done}=await rd.read();if(done)break;buf+=dc.decode(value,{stream:true});
let i;while((i=buf.indexOf('\n\n'))!==-1){const blk=buf.slice(0,i);buf=buf.slice(i+2);
let ev='message',d='';for(const ln of blk.split('\n')){if(ln.startsWith('event:'))ev=ln.slice(6).trim();else if(ln.startsWith('data:'))d+=ln.slice(5).trim()}
if(!d)continue;let pl;try{pl=JSON.parse(d)}catch{continue}
if(ev==='token'){txt+=pl.delta;render(true);dwn()}
else if(ev==='tool_start'){const x=addTool(pl.name,pl.args);x._args=pl.args;tbs.set(pl.id,x);dwn()}
else if(ev==='tool_end'){const x=tbs.get(pl.id);if(x)finishTool(x,pl.result);dwn()}
else if(ev==='media'){mediaCount++;appendMedia(medEl,pl);dwn()}
else if(ev==='error'){streamError=true;b.innerHTML='<span style="color:#ffb3bd">⚠️ '+esc(pl.message)+'</span>';dwn()}
else if(ev==='done'){if(pl.conversation_id&&pl.conversation_id!==cid){cid=pl.conversation_id;markA();loadConvs()}
$('#ct').textContent=p.slice(0,60)}}}
if(!streamError){if(txt.trim()||mediaCount)render(false);else txtEl.innerHTML=rmd('*(no response)*')}
hl(b);dwn();loadConvs();markA();
}catch(e){b.innerHTML='<span style="color:#ffb3bd">⚠️ '+esc(e.message)+'</span>';toast(e.message,1)}
finally{streaming=false;$('#send').disabled=false;$('#ip').focus()}}
async function openConv(id,title){cid=id;$('#ct').textContent=title||'Chat';markA();clearTh();
try{const d=await(await jf('/api/conversations/'+id)).json();
if(!d.messages.length){welcome();return}
for(const m of d.messages){if(m.role==='user')addMsg('user',m.content);
else if(m.role==='assistant'&&m.content)addMsg('assistant',m.content);
else if(m.role==='tool'){const x=addTool(m.name||'tool',{});finishTool(x,m.content)}}
dwn()}catch(e){toast(e.message,1)}}
function newChat(){cid=null;markA();welcome();$('#ip').focus()}
function grow(){const t=$('#ip');t.style.height='auto';t.style.height=Math.min(t.scrollHeight,180)+'px'}
$('#ip').addEventListener('input',grow);
$('#ip').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('#send').onclick=send;$('#nc').onclick=newChat;$('#ts').onclick=()=>$('#sb').classList.toggle('hide');
$('#toolsBtn').onclick=()=>{$('#to').style.display='grid';loadMcpStatus()};
$('#toClose').onclick=()=>{$('#to').style.display='none'};
$('#to').onclick=e=>{if(e.target.id==='to')$('#to').style.display='none'};
$('#lo').onclick=async()=>{try{await fetch(API+'/api/auth/logout',{method:'POST',credentials:'include',headers:H})}catch{}cid=null;authShow()};
let amode='login';
function authShow(){$('#ao').classList.add('show')}
function authHide(){$('#ao').classList.remove('show')}
function setMode(m){amode=m;
$('#at').textContent=m==='login'?'Sign in to Yiz AI':'Create your account';
$('#as').textContent=m==='login'?'Your chats are private to your account.':'Passwords are hashed with Argon2id.';
$('#atog').textContent=m==='login'?'Create account':'Have an account?';
$('#asub').textContent=m==='login'?'Sign in':'Create account';
$('#anf').style.display=m==='register'?'block':'none';
$('#aerr').textContent=''}
async function authSubmit(){const e=$('#ae').value.trim(),p=$('#ap').value,n=$('#an').value.trim(),er=$('#aerr');er.textContent='';
if(!e||!p){er.textContent='email and password required';return}
if(amode==='register'&&p.length<8){er.textContent='password must be at least 8 characters';return}
const path=amode==='login'?'/api/auth/login':'/api/auth/register';
const body=amode==='login'?{email:e,password:p}:{email:e,password:p,name:n||null};
try{const r=await fetch(API+path,{method:'POST',credentials:'include',headers:H,body:JSON.stringify(body)});
const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||d.message||('HTTP '+r.status));
authHide();boot()}catch(err){er.textContent=String(err.message||err)}}
$('#atog').onclick=()=>setMode(amode==='login'?'register':'login');
$('#asub').onclick=authSubmit;
$('#ap').addEventListener('keydown',e=>{if(e.key==='Enter')authSubmit()});
$('#ae').addEventListener('keydown',e=>{if(e.key==='Enter')$('#ap').focus()});
async function loadProviders(){try{const d=await(await fetch(API+'/api/auth/providers')).json();
const box=$('#oap');box.innerHTML='';
for(const p of d.providers){const a=document.createElement('a');
a.href=API+'/api/auth/oauth/'+p.name+'/start?next='+encodeURIComponent(location.pathname);
a.textContent='Continue with '+p.label;box.appendChild(a)}
if(!d.providers.length)box.style.display='none'}catch{}}
async function boot(){try{const r=await fetch(API+'/api/auth/me',{credentials:'include',headers:H});
if(!r.ok){authShow();return}
const me=await r.json();$('#st').textContent=me.email||'signed in';authHide();
await loadConvs();welcome();health();loadModels()}
catch{authShow()}}
setMode('login');loadProviders();boot();
grow();$('#ip').focus();
setInterval(health,30000);
setInterval(async()=>{try{await fetch(API+'/api/auth/refresh',{method:'POST',credentials:'include',headers:H})}catch{}},10*60*1000);
})();
</script></body></html>"""


# ---------------- FASTAPI ----------------
async def retention_worker() -> None:
    """Delete expired sessions, orphaned refresh tokens, and old code runs."""
    while True:
        try:
            async with db.pool.acquire() as c:
                await c.execute(
                    "DELETE FROM sessions WHERE expires_at < NOW() - INTERVAL '30 days'")
                await c.execute(
                    "DELETE FROM refresh_tokens WHERE session_id NOT IN "
                    "(SELECT id FROM sessions)")
                await c.execute(
                    "DELETE FROM code_runs WHERE created_at < NOW() - INTERVAL '30 days'")
            log.info("retention sweep complete")
        except Exception as e:
            log.warning("retention sweep failed: %s", e)
        await asyncio.sleep(6 * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open shared resources on boot and release them on shutdown."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    await db.connect()
    await rate_limiter.init()
    plugin_loader.load_all(reserved_names=frozenset(tools.schema_names()))
    for p in plugin_loader.loaded:
        tools.register(p["name"], p["description"], p["parameters"],
                       p["run"],
                       is_async=inspect.iscoroutinefunction(p["run"]))
    retention_task = asyncio.create_task(retention_worker())
    yield
    retention_task.cancel()
    try:
        await retention_task
    except asyncio.CancelledError:
        pass
    await rate_limiter.close()
    await db.close()


app = FastAPI(title="Yiz AI", version="5.0.0", lifespan=lifespan)

def _cors_origins() -> list[str]:
    """Explicit origin allow-list, derived from config when not set directly."""
    if ALLOWED_ORIGINS and ALLOWED_ORIGINS != ["*"]:
        return ALLOWED_ORIGINS
    derived = [u for u in (FRONTEND_URL, PUBLIC_BASE_URL) if u]
    return list(dict.fromkeys(derived))


if not IS_PROD and ALLOWED_ORIGINS in ([], ["*"]):
    # Development only: any origin, so a local frontend can talk to the API.
    app.add_middleware(CORSMiddleware, allow_origin_regex=".*",
                       allow_credentials=True, allow_methods=["*"],
                       allow_headers=["*"])
elif ALLOWED_ORIGINS == ["*"] and IS_PROD:
    # Wildcard plus cookies is unsafe; serve the allow-list instead and warn.
    log.warning('ALLOWED_ORIGINS="*" is ignored in production; set real origins.')
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins(),
                       allow_credentials=True, allow_methods=["*"],
                       allow_headers=["*"])
else:
    origins = _cors_origins()
    if not origins:
        log.warning("ALLOWED_ORIGINS not set: cross-origin requests are blocked "
                    "(same-origin UI still works).")
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_credentials=True, allow_methods=["*"],
                       allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next: Callable) -> Response:
    """Attach conservative security headers to every response."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("X-XSS-Protection", "0")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("X-Robots-Tag", "noindex, nofollow")
    return response


class RegisterBody(BaseModel):
    """Payload for account registration."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    name: Optional[str] = Field(default=None, max_length=80)


class LoginBody(BaseModel):
    """Payload for password login."""
    email: EmailStr
    password: str


class ChatRequest(BaseModel):
    """Payload for a plain chat turn."""
    prompt: str = Field(min_length=1, max_length=20000)
    conversation_id: Optional[str] = None
    model: Optional[str] = Field(default=None, max_length=200)


class ChatWithFilesRequest(BaseModel):
    """Payload for a chat turn that references uploaded attachments."""
    prompt: str = Field(min_length=1, max_length=20000)
    conversation_id: Optional[str] = None
    attachment_ids: list[str] = Field(default_factory=list, max_length=5)
    model: Optional[str] = Field(default=None, max_length=200)


async def _optional_user(request: Request) -> Optional[dict]:
    """Resolve the caller if a valid session is present, else None."""
    token = request.cookies.get(COOKIE_ACCESS)
    auth = request.headers.get("authorization")
    if auth and auth.startswith("Bearer "):
        token = auth[7:].strip()
    if not token:
        return None
    sess = await _session_from_token(token)
    if not sess:
        return None
    return await db.get_user(sess["user_id"])


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness endpoint: always HTTP 200 while the web process is up."""
    info: dict[str, Any] = {"ok": True, "service": "yiz-ai", "version": "5.0.0",
                            "sandbox": SANDBOX_PROVIDER,
                            "providers": configured_providers(),
                            "mcp_servers": mcp_client.list_servers()}
    try:
        provider, model = pick_provider()
        info["provider"], info["model"] = provider, model
    except LLMError as exc:
        info["provider"], info["model"] = None, None
        info["warning"] = str(exc)
    return info


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Alias of /api/health for platforms that default to /healthz."""
    return await health()


@app.get("/api/models")
async def models(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Every model reachable with the configured provider credentials."""
    await rate_limiter.hit(f"models:{user['id']}", RL_API_PER_MIN, 60)
    return await list_all_models()


@app.get("/api/mcp/status")
async def mcp_status(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Configured MCP servers, their reachable tools, and any rejected entries.

    Backs the sidebar's Tools & MCP panel — this is what makes MCP visible in
    the UI instead of only existing as tools the model can silently call.
    """
    await rate_limiter.hit(f"mcpstatus:{user['id']}", RL_API_PER_MIN, 60)
    servers = []
    for name in mcp_client.list_servers():
        mcp_tools = await mcp_client.list_tools(name)
        servers.append({"name": name, "url": mcp_client.servers[name],
                        "status": "ok" if mcp_tools else "unreachable",
                        "tools": [{"name": t["name"], "description": t["description"]}
                                 for t in mcp_tools]})
    rejected = [{"name": n, "url": u, "reason": "rejected by URL validation "
                "(see server logs)"} for n, u in mcp_client.rejected.items()]
    return {"servers": servers, "rejected": rejected,
            "builtin_tool_count": len(tools.schemas())}


@app.get("/api/ready")
async def ready() -> Response:
    """Readiness endpoint: DB and an LLM provider must be configured."""
    if db.pool is None:
        return Response(
            content=json.dumps({"ok": False, "error": "database not connected"}),
            status_code=503, media_type="application/json")
    try:
        provider, model = pick_provider()
    except LLMError as exc:
        return Response(
            content=json.dumps({"ok": False, "error": str(exc)}),
            status_code=503, media_type="application/json")
    return Response(
        content=json.dumps({"ok": True, "provider": provider, "model": model}),
        status_code=200, media_type="application/json")


@app.get("/api/auth/providers")
async def providers() -> dict[str, Any]:
    """List the OAuth providers that are fully configured."""
    return {"providers": [{"name": n, "label": "Google" if n == "google" else "GitHub"}
                          for n in oauth_providers if oauth_providers[n].enabled]}


@app.post("/api/auth/register")
async def register(body: RegisterBody, request: Request, response: Response) -> dict:
    """Create an account and start a session."""
    if not ALLOW_REGISTRATION:
        raise HTTPException(403, "registration disabled")
    ip = request.client.host if request.client else "?"
    await rate_limiter.hit(f"signup:{ip}", RL_SIGNUP_PER_MIN, 60)
    if not EMAIL_RE.match(body.email):
        raise HTTPException(400, "invalid email")
    if await db.get_user_by_email(body.email):
        raise HTTPException(409, "email already registered")
    user = await db.create_user(body.email, hash_password(body.password), body.name)
    return await _issue_session(user, request, response)


@app.post("/api/auth/login")
async def login(body: LoginBody, request: Request, response: Response) -> dict:
    """Authenticate with email and password."""
    ip = request.client.host if request.client else "?"
    await rate_limiter.hit(f"login:{ip}", RL_LOGIN_PER_MIN, 60)
    user = await db.get_user_by_email(body.email)
    if not user or not user.get("password_hash"):
        verify_password(_DUMMY_HASH, body.password)
        raise HTTPException(401, "invalid email or password")
    if not verify_password(user["password_hash"], body.password):
        raise HTTPException(401, "invalid email or password")
    return await _issue_session(user, request, response)


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response,
                 yiz_access: Optional[str] = Cookie(default=None)) -> dict[str, bool]:
    """Revoke the current session and clear its cookies."""
    if yiz_access:
        sess = await _session_from_token(yiz_access)
        if sess:
            await db.revoke_session(sess["session_id"])
    clear_auth_cookies(response)
    return {"ok": True}


@app.post("/api/auth/logout-all")
async def logout_all(response: Response, user: dict = Depends(get_current_user)) -> dict[str, int]:
    """Revoke every session belonging to the caller."""
    n = await db.revoke_all_sessions(user["id"])
    clear_auth_cookies(response)
    return {"revoked": n}


@app.post("/api/auth/refresh")
async def refresh(response: Response,
                  yiz_refresh: Optional[str] = Cookie(default=None)
                  ) -> dict[str, bool]:
    """Rotate the refresh token, detecting reuse."""
    if not yiz_refresh:
        raise HTTPException(401, "no refresh token")
    hashed = hash_refresh(yiz_refresh)
    row = await db.get_refresh(hashed)
    if not row:
        raise HTTPException(401, "unknown refresh token")
    if row["used_at"] is not None:
        await db.revoke_session(row["session_id"])
        clear_auth_cookies(response)
        raise HTTPException(401, "refresh token reuse detected")
    exp = row["expires_at"]
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp < datetime.now(timezone.utc):
        raise HTTPException(401, "refresh token expired")
    sess = await db.get_session(row["session_id"])
    if not sess or sess["revoked"]:
        raise HTTPException(401, "session revoked")
    user = await db.get_user(sess["user_id"])
    if not user or not user["is_active"]:
        raise HTTPException(401, "user not found")
    raw, new_hash = new_refresh_token()
    consumed = await db.mark_refresh_used(hashed, new_hash)
    if not consumed:
        await db.revoke_session(sess["id"])
        clear_auth_cookies(response)
        raise HTTPException(401, "refresh token reuse detected")
    await db.store_refresh(sess["id"], new_hash, REFRESH_TTL_DAYS)
    set_auth_cookies(response, make_access_token(user["id"], sess["id"]), raw)
    return {"ok": True}


@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Return the authenticated caller."""
    return {"id": user["id"], "email": user["email"], "name": user["name"]}


@app.get("/api/auth/oauth/{provider}/start")
async def oauth_start(provider: str,
                      next: Optional[str] = Query(default=None)
                      ) -> RedirectResponse:
    """Redirect the caller to an OAuth provider."""
    p = oauth_providers.get(provider)
    if not p or not p.enabled:
        raise HTTPException(404, "provider not configured")
    return RedirectResponse(p.authorize(_sign_state(provider, _safe_next(next))),
                            status_code=302)


@app.get("/api/auth/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request,
                         code: Optional[str] = Query(default=None),
                         state: Optional[str] = Query(default=None),
                         error: Optional[str] = Query(default=None)) -> RedirectResponse:
    """Complete an OAuth exchange and start a session."""
    p = oauth_providers.get(provider)
    if not p or not p.enabled:
        raise HTTPException(404, "provider not configured")
    if error:
        raise HTTPException(400, f"provider error: {error}")
    if not code or not state:
        raise HTTPException(400, "missing code or state")
    claims = _verify_state(state)
    if claims.get("p") != provider:
        raise HTTPException(400, "state mismatch")
    tokens = await p.exchange(code)
    ident = await p.fetch_identity(tokens)
    user = await _upsert_oauth_user(provider, ident)
    resp = RedirectResponse(_safe_next(claims.get("n")), status_code=302)
    await _issue_session(user, request, resp)
    return resp


@app.get("/api/conversations")
async def list_convs(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """List the caller's conversations."""
    await rate_limiter.hit(f"api:{user['id']}", RL_API_PER_MIN, 60)
    return {"conversations": await db.list_conversations(user["id"])}


@app.post("/api/conversations")
async def new_conv(user: dict = Depends(get_current_user)) -> dict[str, str]:
    """Create an empty conversation."""
    return {"id": await db.create_conversation(user["id"])}


@app.get("/api/conversations/{cid}")
async def get_conv(cid: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Return one conversation and its messages."""
    conv = await db.get_conversation(cid, user["id"])
    if not conv:
        raise HTTPException(404, "not found")
    return {"id": cid, "title": conv["title"],
            "messages": await db.get_messages(cid)}


@app.delete("/api/conversations/{cid}")
async def del_conv(cid: str, user: dict = Depends(get_current_user)) -> dict[str, str]:
    """Delete one of the caller's conversations."""
    if not await db.delete_conversation(cid, user["id"]):
        raise HTTPException(404, "not found")
    return {"deleted": cid}


@app.get("/api/tools")
async def list_tools(_: Any = Depends(get_current_user)) -> dict[str, Any]:
    """List the tool schemas exposed to the model."""
    return {"tools": tools.schemas()}


@app.post("/api/upload")
async def upload(file: UploadFile = FastFile(...),
                 user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Accept a multipart upload, store it, and record the attachment."""
    data = await file.read()
    if len(data) > MEDIA_MAX_BYTES:
        raise HTTPException(413, f"file exceeds {MEDIA_MAX_BYTES} bytes")
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename or "upload.bin")[:150]
    ext = Path(safe_name).suffix.lstrip(".").lower() or "bin"
    token = CURRENT_USER_ID.set(user["id"])
    try:
        stored = await store_media(data, ext, file.content_type or "application/octet-stream", safe_name)
    finally:
        CURRENT_USER_ID.reset(token)
    aid = await db.add_attachment(user["id"], safe_name, stored,
                                  file.content_type or "application/octet-stream",
                                  len(data))
    return {"id": aid, "filename": safe_name, "stored_name": stored,
            "size": len(data), "content_type": file.content_type,
            "url": f"/api/media/{stored}?name={quote(safe_name)}"}


@app.get("/api/attachments")
async def list_attachments(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """List the caller's uploaded attachments."""
    return {"attachments": await db.list_attachments(user["id"])}


@app.get("/api/media/{stored:path}")
async def serve_media(stored: str, name: Optional[str] = Query(default=None),
                      user: dict = Depends(get_current_user)) -> FileResponse:
    """Stream a stored media file only to its owning user."""
    try:
        key = media_store._safe_key(stored)
    except ValueError:
        raise HTTPException(404, "not found")
    meta = await db.get_media(key, user["id"])
    if not meta:
        raise HTTPException(404, "not found")
    p = media_store.path(key)
    if not p:
        raise HTTPException(404, "not found")
    safe = _safe_name(name or meta.get("original_name"), "bin") if (name or meta.get("original_name")) else None
    ctype = meta.get("content_type") or mimetypes.guess_type(safe or key)[0] or "application/octet-stream"
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"}
    if safe:
        headers["Content-Disposition"] = f'inline; filename="{safe}"'
    return FileResponse(p, media_type=ctype, headers=headers)


def sse(event: str, data: Any) -> str:
    """Format one server-sent event frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def with_heartbeat(agen: AsyncGenerator[str, None],
                         interval: float = 15.0) -> AsyncIterator[str]:
    """Yield from agen, emitting an SSE comment ping while it is idle."""
    aiter = agen.__aiter__()
    pending: Optional[asyncio.Task] = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(aiter.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield ": ping\n\n"
                continue
            task, pending = pending, None
            try:
                yield task.result()
            except StopAsyncIteration:
                return
    except asyncio.CancelledError:
        raise
    finally:
        if pending is not None:
            pending.cancel()
        await agen.aclose()


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep the most recent turns, never orphaning a tool result."""
    if len(history) <= MAX_CONTEXT_MESSAGES:
        return history
    window = history[-MAX_CONTEXT_MESSAGES:]
    while window and window[0].get("role") == "tool":
        window.pop(0)
    return window


HEAVY_WORKER_TOOLS = {
    "generate_image", "generate_video", "text_to_speech",
    "generate_text_file", "generate_csv", "generate_json_file", "generate_markdown_file",
    "generate_pdf", "generate_xlsx", "generate_docx", "make_chart", "generate_qr",
}


async def dispatch_tool(name: str, args: dict[str, Any], user_id: str) -> str:
    """Run heavy media work in Redis worker when available, otherwise locally."""
    if name in HEAVY_WORKER_TOOLS and REDIS_URL:
        try:
            from job_queue import enqueue_and_wait, worker_available
        except ImportError:
            log.debug("job_queue module not deployed; running %s in-process", name)
        else:
            try:
                if await worker_available():
                    result = await enqueue_and_wait(name, args, user_id=user_id)
                    return json.dumps(result, default=str)
                log.warning("no fresh worker heartbeat; executing %s locally", name)
            except Exception as e:
                log.warning("worker dispatch failed for %s (%s); running locally", name, e)
    return await tools.call(name, args, user_id=user_id)


async def agent_stream(prompt: str, cid: str, user_id: str,
                       model: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Run the tool-calling loop for one turn, emitting SSE frames."""
    await db.add_message(cid, "user", prompt)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += _trim_history(await db.get_messages(cid))

    schemas = tools.schemas()
    for m in await mcp_client.all_tools():
        schemas.append({"name": m["name"],
                        "description": f"[MCP:{m['server']}] {m['description']}",
                        "parameters": m["input_schema"]})

    final_text = ""
    for _ in range(MAX_STEPS):
        parts, calls = [], []
        try:
            async for ev in stream_chat(msgs, schemas, model, prompt):
                if ev["type"] == "text":
                    parts.append(ev["delta"])
                    yield sse("token", {"delta": ev["delta"]})
                elif ev["type"] == "tool_calls":
                    calls = ev["calls"]
        except LLMError as e:
            yield sse("error", {"message": str(e)})
            return
        except Exception as e:
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})
            return

        text = "".join(parts)
        if not calls:
            final_text = text
            break

        await db.add_message(cid, "assistant", text, tool_calls=calls)
        msgs.append({"role": "assistant", "content": text, "tool_calls": calls})

        for call in calls:
            yield sse("tool_start", {"id": call["id"], "name": call["name"],
                                     "args": call["args"]})
            try:
                if call["name"].startswith("mcp__"):
                    _, server, tool = call["name"].split("__", 2)
                    result = json.dumps(await mcp_client.call_tool(server, tool, call["args"]))
                else:
                    token = CURRENT_USER_ID.set(user_id)
                    try:
                        result = await dispatch_tool(call["name"], call["args"], user_id)
                    finally:
                        CURRENT_USER_ID.reset(token)
            except Exception as exc:
                log.exception("tool %s failed during agent loop", call["name"])
                result = json.dumps({"error": f"tool execution failed: {type(exc).__name__}: {exc}"})
            # Keep model context bounded even if an external/MCP tool returns a huge payload.
            result = result[:20000]

            media_event = None
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    for key, kind in (("image_url", "image"),
                                      ("audio_url", "audio"),
                                      ("video_url", "video"),
                                      ("download_url", "file")):
                        if key in parsed:
                            media_event = {"type": kind, "url": parsed[key],
                                           "name": parsed.get("filename")}
                            break
            except Exception:
                pass

            yield sse("tool_end", {"id": call["id"], "name": call["name"],
                                   "result": result[:4000]})
            if media_event:
                yield sse("media", {"id": call["id"], "type": media_event["type"],
                                    "url": media_event["url"],
                                    "name": media_event.get("name")})
            await db.add_message(cid, "tool", result,
                                 tool_call_id=call["id"], name=call["name"])
            msgs.append({"role": "tool", "content": result,
                         "tool_call_id": call["id"], "name": call["name"]})
    else:
        final_text = "(reached max tool steps)"

    if final_text:
        await db.add_message(cid, "assistant", final_text)
    yield sse("done", {"conversation_id": cid})


def _stream_headers() -> dict[str, str]:
    """Headers that keep SSE responses unbuffered end to end."""
    return {"Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"}


@app.post("/api/chat")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)) -> StreamingResponse:
    """Stream one chat turn."""
    await rate_limiter.hit(f"chat:{user['id']}", RL_API_PER_MIN, 60)
    cid = body.conversation_id
    if not cid:
        cid = await db.create_conversation(user["id"], body.prompt[:60])
    else:
        if not await db.get_conversation(cid, user["id"]):
            raise HTTPException(404, "conversation not found")
        if not await db.get_messages(cid):
            await db.rename_conversation(cid, user["id"], body.prompt[:60])
    return StreamingResponse(
        with_heartbeat(agent_stream(body.prompt, cid, user["id"], body.model)),
        media_type="text/event-stream", headers=_stream_headers())


@app.post("/api/chat-with-files")
async def chat_with_files(body: ChatWithFilesRequest,
                          user: dict = Depends(get_current_user)) -> StreamingResponse:
    """Stream one chat turn with attachment context prepended."""
    await rate_limiter.hit(f"chat:{user['id']}", RL_API_PER_MIN, 60)
    context_parts = []
    for aid in body.attachment_ids[:5]:
        row = await db.get_attachment(aid, user["id"])
        if not row:
            continue
        p = media_store.path(row["stored_name"])
        if not p:
            continue
        ct = row["content_type"] or ""
        if ct.startswith("text/") or ct in ("application/json", "application/csv"):
            text = p.read_text(errors="replace")[:40000]
            context_parts.append(f"[Attached file: {row['filename']}]\n{text}")
        elif ct.startswith("audio/"):
            result = await _transcribe_audio(row["stored_name"], user["id"])
            if result.get("text"):
                context_parts.append(
                    f"[Attached voice note: {row['filename']}]\n{result['text']}")
            else:
                context_parts.append(
                    f"[Attached voice note: {row['filename']} ({ct}, "
                    f"{p.stat().st_size} bytes) — transcription unavailable "
                    f"({result.get('error', 'unknown error')}); call "
                    f"transcribe_audio with source '{row['stored_name']}' to retry]")
        else:
            extracted = _extract_attachment_text(p, ct)
            if extracted:
                context_parts.append(
                    f"[Attached file: {row['filename']}]\n{extracted}")
            else:
                context_parts.append(
                    f"[Attached file: {row['filename']} ({ct}, "
                    f"{p.stat().st_size} bytes) — to analyze, call analyze_data "
                    f"or query_data with source '{row['stored_name']}']")

    full_prompt = body.prompt
    if context_parts:
        full_prompt = "\n\n".join(context_parts) + "\n\nUser: " + body.prompt

    cid = body.conversation_id
    if not cid:
        cid = await db.create_conversation(user["id"], body.prompt[:60])
    else:
        if not await db.get_conversation(cid, user["id"]):
            raise HTTPException(404, "conversation not found")
        if not await db.get_messages(cid):
            await db.rename_conversation(cid, user["id"], body.prompt[:60])

    return StreamingResponse(
        with_heartbeat(agent_stream(full_prompt, cid, user["id"], body.model)),
        media_type="text/event-stream", headers=_stream_headers())


@app.get("/", response_class=HTMLResponse)
async def index() -> Response:
    """Serve the embedded frontend in single-service mode or redirect to the standalone UI."""
    if FRONTEND_URL:
        return RedirectResponse(FRONTEND_URL)
    return HTMLResponse(FRONTEND_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("yiz_ai:app", host="0.0.0.0", port=PORT, reload=not IS_PROD)
