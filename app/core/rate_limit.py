import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 20, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Only rate limit AI and RAG chat endpoints
        path = request.url.path
        if request.method == "POST" and ("/ai/" in path or "/rag/chat" in path):
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            window_start = now - self.window_seconds

            # Clean old entries
            self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]

            # Check limit
            if len(self.requests[client_ip]) >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Limit: {self.max_requests} requests per {self.window_seconds}s. Try again later."
                )

            self.requests[client_ip].append(now)

        return await call_next(request)
