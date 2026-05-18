import json
import hashlib
from typing import Optional, Any, Callable
from app.core.config import settings

_client = None

def get_redis():
    global _client
    if _client is None:
        try:
            import redis
            _client = redis.from_url(
                settings.REDIS_CONNECTION_URL,
                socket_timeout=3,
                socket_connect_timeout=3,
                decode_responses=True,
                max_connections=10,
            )
            _client.ping()
        except Exception:
            _client = None
    return _client

def _safe_serialize(obj):
    """Return a JSON-safe representation, skipping non-serializable objects."""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return str(type(obj).__name__)

def cache_key(prefix: str, *args, **kwargs) -> str:
    safe_args = [_safe_serialize(a) for a in args if not hasattr(a, 'execute')]
    safe_kwargs = {k: _safe_serialize(v) for k, v in kwargs.items() if k != 'db'}
    raw = f"{prefix}:{safe_args}:{safe_kwargs}"
    return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"

def cached(ttl: int = 300):
    def decorator(func: Callable):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            r = get_redis()
            if r is None:
                return func(*args, **kwargs)
            key = cache_key(func.__name__)
            try:
                cached = r.get(key)
                if cached is not None:
                    return json.loads(cached)
            except Exception:
                pass
            result = func(*args, **kwargs)
            try:
                r.setex(key, ttl, json.dumps(result, default=str))
                _track_key(r, func.__name__, key)
            except Exception:
                pass
            return result
        return wrapper
    return decorator

def _track_key(r, prefix: str, key: str):
    set_key = f"cache_keys:{prefix}"
    r.sadd(set_key, key)
    r.expire(set_key, 86400)

def invalidate_cache(prefix: str):
    r = get_redis()
    if r is None:
        return
    try:
        set_key = f"cache_keys:{prefix}"
        keys = r.smembers(set_key)
        if keys:
            r.delete(*keys)
            r.delete(set_key)
        else:
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
