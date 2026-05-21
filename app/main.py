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
            if re.search(r'/public/(projects|skills|experiences|education|services)', path):
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

# ── Register routers (all heavy deps are now lazy-imported inside services) ──

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

@app.get("/")
def root():
    return {"message": "Welcome to the Portfolio API Modular Monolith!"}

@app.get("/health")
@app.get("/kaithhealthcheck")
@app.get("/kaithheathcheck")
def health_check():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}...")

    # Register AI agents
    try:
        from app.agents import register_all_agents
        register_all_agents()
        print("  ✓ Agents: Registered")
    except Exception as e:
        print(f"  ⚠ Agents: Failed to register: {e}")
    
    # Database migration: Alter Experience columns start_date and end_date from DATE to VARCHAR
    try:
        from sqlalchemy import text
        from app.core.database import engine
        with engine.begin() as conn:
            print("  🔨 Running DB migration: Altering experience date columns to VARCHAR...")
            conn.execute(text("ALTER TABLE public_experiences ALTER COLUMN start_date TYPE VARCHAR(255) USING start_date::varchar;"))
            conn.execute(text("ALTER TABLE public_experiences ALTER COLUMN end_date TYPE VARCHAR(255) USING end_date::varchar;"))
            print("  ✓ DB migration: Completed successfully")
    except Exception as db_err:
        print(f"  ⚠ DB migration: Failed (this might be normal if already converted): {db_err}")

    try:
        from app.modules.github_trends.tasks import setup_github_trends_scheduler
        setup_github_trends_scheduler()
        print("  ✓ Scheduler: Started")
    except Exception as e:
        print(f"   Scheduler: Failed to start: {e}")

    # RAG Knowledge Indexing — only on local/dev (skip on Vercel serverless)
    _is_serverless = bool(__import__('os').environ.get("VERCEL") or __import__('os').environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    if not _is_serverless:
        try:
            from app.modules.rag import pipeline, vector_store
            vector_store.ensure_collection()
            result = pipeline.index_knowledge()
            if result.get("status") == "success":
                print(f"  ✓ RAG: Indexed {result.get('points_indexed', 0)} chunks")
            else:
                print(f"  ⚠ RAG: Indexing returned {result.get('status')}")
        except Exception as e:
            print(f"  ⚠ RAG: Indexing failed (run seed_data.py first): {e}")
    else:
        print("  ⏭ RAG: Skipped on serverless (use /api/v1/rag/index endpoint)")

    print("✅ Startup complete")

