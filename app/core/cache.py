import json
import hashlib
from typing import Optional, Any, Callable
import redis
from app.core.config import settings

_client: Optional[redis.Redis] = None

def get_redis() -> Optional[redis.Redis]:
    global _client
    if _client is None:
        try:
            _client = redis.from_url(
                settings.REDIS_CONNECTION_URL,
                socket_timeout=3,
                socket_connect_timeout=3,
                decode_responses=True,
            )
            _client.ping()
        except Exception:
            _client = None
    return _client

def cache_key(prefix: str, *args, **kwargs) -> str:
    raw = f"{prefix}:{json.dumps(args)}:{json.dumps(kwargs, sort_keys=True)}"
    return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"

def cached(ttl: int = 300):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            r = get_redis()
            if r is None:
                return func(*args, **kwargs)
            key = cache_key(func.__name__, *args, **kwargs)
            try:
                cached = r.get(key)
                if cached is not None:
                    return json.loads(cached)
            except Exception:
                pass
            result = func(*args, **kwargs)
            try:
                r.setex(key, ttl, json.dumps(result, default=str))
            except Exception:
                pass
            return result
        return wrapper
    return decorator

def invalidate_cache(prefix: str):
    r = get_redis()
    if r is None:
        return
    try:
        for key in r.scan_iter(f"{prefix}:*"):
            r.delete(key)
    except Exception:
        pass

def get_cached_or_set(key: str, ttl: int, fallback: Callable) -> Any:
    r = get_redis()
    if r is not None:
        try:
            cached = r.get(key)
            if cached is not None:
                return json.loads(cached)
        except Exception:
            pass
    result = fallback()
    if r is not None:
        try:
            r.setex(key, ttl, json.dumps(result, default=str))
        except Exception:
            pass
    return result
