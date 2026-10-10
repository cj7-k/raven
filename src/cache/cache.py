"""
src/cache/cache.py

Answer cache for RAVEN. A repeated question skips retrieval AND the LLM call
entirely, which is the biggest latency win in the whole pipeline.

Backends:
  - Redis (preferred): shared, survives server restarts, used in Docker.
  - In-memory (automatic fallback): used when Redis isn't reachable, so the
    app keeps working on a laptop with no Redis installed. Lives and dies
    with the server process.

Behaviour worth knowing:
  - Questions are normalised before lookup, so "What causes eczema?" and
    "what causes  eczema" share one cache entry.
  - Entries expire after CACHE_TTL_SECONDS (default 1 hour).
  - If Redis dies mid-run, requests keep working (they just stop being cached
    in Redis and use the in-memory fallback until it comes back).
  - After you re-scrape / re-embed new data, clear the cache so old answers
    aren't served:   python src/cache/cache.py --clear

Config (environment variables):
    REDIS_URL           default redis://localhost:6379/0
    CACHE_TTL_SECONDS   default 3600

CLI:
    python src/cache/cache.py --info     # which backend, how many entries
    python src/cache/cache.py --clear    # delete all cached answers
"""

import argparse
import hashlib
import json
import os
import re
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_TTL_SECONDS = 3600
KEY_PREFIX = "raven:answer:"
MAX_MEMORY_ENTRIES = 500


# ---------------------------------------------------------------------------
# Key building
# ---------------------------------------------------------------------------

def normalize_question(question: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation."""
    q = re.sub(r"\s+", " ", question.strip().lower())
    return q.rstrip("?!. ")


def make_key(question: str, top_k: int) -> str:
    raw = f"{normalize_question(question)}|{top_k}"
    return KEY_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class AnswerCache:
    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: Optional[int] = None, client=None):
        self.ttl = int(ttl_seconds or os.getenv("CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS))
        self.hits = 0
        self.misses = 0
        self._memory = {}          # key -> (expires_at, value)
        self._warned = False

        if client is not None:     # injected client (used by tests)
            self._redis = client
        else:
            self._redis = self._try_connect(redis_url or os.getenv("REDIS_URL", DEFAULT_REDIS_URL))

    # -- connection --------------------------------------------------------

    @staticmethod
    def _try_connect(url: str):
        try:
            import redis
            client = redis.Redis.from_url(
                url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1
            )
            client.ping()
            return client
        except Exception:
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def _warn_once(self, error: Exception):
        if not self._warned:
            print(f"[cache] Redis error ({error}); falling back to in-memory cache for now.")
            self._warned = True

    # -- in-memory helpers -------------------------------------------------

    def _memory_get(self, key: str):
        entry = self._memory.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._memory.pop(key, None)
            return None
        return value

    def _memory_set(self, key: str, value: dict):
        if len(self._memory) >= MAX_MEMORY_ENTRIES:
            self._memory.pop(next(iter(self._memory)))   # evict oldest
        self._memory[key] = (time.monotonic() + self.ttl, value)

    # -- public API --------------------------------------------------------

    def get(self, question: str, top_k: int) -> Optional[dict]:
        key = make_key(question, top_k)
        value = None

        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                value = json.loads(raw) if raw else None
            except Exception as e:
                self._warn_once(e)
                value = self._memory_get(key)
        else:
            value = self._memory_get(key)

        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, question: str, top_k: int, value: dict) -> None:
        key = make_key(question, top_k)

        if self._redis is not None:
            try:
                self._redis.setex(key, self.ttl, json.dumps(value))
                return
            except Exception as e:
                self._warn_once(e)
        self._memory_set(key, value)

    def entry_count(self) -> int:
        if self._redis is not None:
            try:
                return sum(1 for _ in self._redis.scan_iter(match=KEY_PREFIX + "*"))
            except Exception as e:
                self._warn_once(e)
        return len(self._memory)

    def clear(self) -> int:
        deleted = len(self._memory)
        self._memory.clear()
        if self._redis is not None:
            try:
                keys = list(self._redis.scan_iter(match=KEY_PREFIX + "*"))
                if keys:
                    self._redis.delete(*keys)
                deleted += len(keys)
            except Exception as e:
                self._warn_once(e)
        return deleted

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "hits": self.hits,
            "misses": self.misses,
            "entries": self.entry_count(),
            "ttl_seconds": self.ttl,
        }


# ---------------------------------------------------------------------------
# Shared instance
# ---------------------------------------------------------------------------

_cache: Optional[AnswerCache] = None


def get_cache() -> AnswerCache:
    global _cache
    if _cache is None:
        _cache = AnswerCache()
        print(f"[cache] using {_cache.backend} backend (ttl={_cache.ttl}s)")
    return _cache


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect or clear the RAVEN answer cache")
    parser.add_argument("--info", action="store_true", help="Show backend and entry count")
    parser.add_argument("--clear", action="store_true", help="Delete all cached answers")
    args = parser.parse_args()

    cache = get_cache()
    if args.clear:
        print(f"Cleared {cache.clear()} cached answer(s).")
        if cache.backend == "memory":
            print("(Redis isn't running, so only this process's in-memory cache was cleared. "
                  "The API server's own memory cache clears when you restart it.)")
    else:
        info = cache.info()
        print(f"Backend: {info['backend']} | entries: {info['entries']} | ttl: {info['ttl_seconds']}s")


if __name__ == "__main__":
    main()
