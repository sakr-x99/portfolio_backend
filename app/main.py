from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import re

from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        if request.method == "GET":
            path = request.url.path
            if re.search(r'/public/(projects|skills|experiences|education|services|articles)', path):
                response.headers["Cache-Control"] = "public, max-age=120, s-maxage=120, stale-while-revalidate=60"
            elif re.search(r'/public/', path):
                response.headers["Cache-Control"] = "public, max-age=60, s-maxage=60"
        return response

# Set all origins enabled by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CacheControlMiddleware)

from app.core.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, max_requests=20, window_seconds=60)

# ── Instant endpoints (available before heavy imports) ──────────────

@app.get("/")
def root():
    return {"message": "Welcome to the Portfolio API Modular Monolith!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── Lazy initialization (runs AFTER uvicorn binds the port) ─────────

def _register_routers():
    """Import and register all routers. Heavy imports happen here."""
    from app.modules.public_portfolio.routers import router as public_router
    from app.modules.biz_management.routers import router as biz_router
    from app.modules.ai.routers import router as ai_router
    from app.modules.rag.routers import router as rag_router
    from app.modules.github_trends.router import router as trends_router
    from app.modules.auth.routers import router as auth_router

    app.include_router(public_router, prefix=f"{settings.API_V1_STR}/public", tags=["Public Portfolio"])
    app.include_router(biz_router, prefix=f"{settings.API_V1_STR}/admin", tags=["Business Management"])
    app.include_router(ai_router, prefix=f"{settings.API_V1_STR}/ai", tags=["AI Integration"])
    app.include_router(rag_router, prefix=f"{settings.API_V1_STR}/rag", tags=["RAG Pipeline"])
    app.include_router(trends_router, prefix=f"{settings.API_V1_STR}/trends", tags=["GitHub Trends"])
    app.include_router(auth_router, prefix=f"{settings.API_V1_STR}/auth", tags=["Auth"])
    print("  ✓ All routers registered")

async def _init_services():
    """Initialize database, cache, vector store, and schedulers."""
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION} service init...")

    # 1. Database
    try:
        from app.core.database import engine, Base
        from sqlalchemy import text
        from app.modules.public_portfolio import models as _
        from app.modules.biz_management import models as __
        from app.modules.github_trends import models as ___
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            for col in ["gradient", "content"]:
                r = conn.execute(text(f"SELECT 1 FROM information_schema.columns WHERE table_name='public_projects' AND column_name='{col}'"))
                if not r.fetchone():
                    col_type = "VARCHAR" if col == "gradient" else "TEXT"
                    conn.execute(text(f"ALTER TABLE public_projects ADD COLUMN {col} {col_type}"))
                    conn.commit()
        print("  ✓ Database: OK")
    except Exception as e:
        print(f"  ⚠ Database: {e}")

    # 2. Redis
    try:
        import redis
        r = redis.from_url(settings.REDIS_CONNECTION_URL, socket_timeout=5)
        r.ping()
        print("  ✓ Redis: OK")
    except Exception as e:
        print(f"  ⚠ Redis: {e}")

    # 3. Qdrant
    try:
        from app.modules.rag.vector_store import ensure_collection
        ensure_collection()
        print("  ✓ Qdrant: OK")
    except Exception as e:
        print(f"  ⚠ Qdrant: {e}")

    # 4. Schedulers
    try:
        from app.modules.github_trends.tasks import setup_github_trends_scheduler
        setup_github_trends_scheduler()
        print("  ✓ Scheduler: OK")
    except Exception as e:
        print(f"  ⚠ Scheduler: {e}")

    print("✅ All services initialized")

async def _lazy_init():
    """Run heavy initialization in background after uvicorn is already listening."""
    import asyncio
    loop = asyncio.get_event_loop()
    # Run heavy imports in thread pool so event loop stays responsive
    await loop.run_in_executor(None, _register_routers)
    # Then init services
    await _init_services()

@app.on_event("startup")
async def startup_event():
    import asyncio
    # Fire-and-forget: uvicorn starts listening IMMEDIATELY
    asyncio.create_task(_lazy_init())
