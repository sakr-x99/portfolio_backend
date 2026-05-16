import time
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.cache import get_redis

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 20, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        is_write = request.method in ("POST", "PUT", "DELETE")
        if is_write and ("/ai/" in path or "/rag/" in path or "/admin/" in path or "/public/" in path):
            client_ip = request.client.host if request.client else "unknown"
            now = int(time.time())
            window_key = f"ratelimit:{client_ip}:{now // self.window_seconds}"

            r = get_redis()
            if r is not None:
                try:
                    count = r.incr(window_key)
                    if count == 1:
                        r.expire(window_key, self.window_seconds)
                    if count > self.max_requests:
                        raise HTTPException(
                            status_code=429,
                            detail=f"Too many requests. Limit: {self.max_requests} per {self.window_seconds}s."
                        )
                except HTTPException:
                    raise
                except Exception:
                    pass
            else:
                pass

        return await call_next(request)
