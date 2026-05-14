from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

from fastapi.middleware.cors import CORSMiddleware

# Set all origins enabled by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.modules.public_portfolio.routers import router as public_router
from app.modules.biz_management.routers import router as biz_router
from app.modules.ai.routers import router as ai_router
from app.modules.rag.routers import router as rag_router
from app.modules.github_trends.router import router as trends_router

app.include_router(public_router, prefix=f"{settings.API_V1_STR}/public", tags=["Public Portfolio"])
app.include_router(biz_router, prefix=f"{settings.API_V1_STR}/admin", tags=["Business Management"])
app.include_router(ai_router, prefix=f"{settings.API_V1_STR}/ai", tags=["AI Integration"])
app.include_router(rag_router, prefix=f"{settings.API_V1_STR}/rag", tags=["RAG Pipeline"])
app.include_router(trends_router, prefix=f"{settings.API_V1_STR}/trends", tags=["GitHub Trends"])

@app.get("/")
def root():
    return {"message": "Welcome to the Portfolio API Modular Monolith!"}

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    
    # 1. Database Initialization & Migrations
    try:
        print("  → Initializing Database & Tables...")
        from app.core.database import engine, Base
        from sqlalchemy import text
        # Import all models to ensure they are registered
        from app.modules.public_portfolio import models as public_models
        from app.modules.biz_management import models as biz_models
        from app.modules.github_trends import models as trends_models
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        # Manual Migrations
        with engine.connect() as conn:
            # Check if gradient exists
            result = conn.execute(text("SELECT 1 FROM information_schema.columns WHERE table_name='public_projects' AND column_name='gradient'"))
            if not result.fetchone():
                conn.execute(text("ALTER TABLE public_projects ADD COLUMN gradient VARCHAR"))
                conn.commit()
                
            # Check if content exists
            result = conn.execute(text("SELECT 1 FROM information_schema.columns WHERE table_name='public_projects' AND column_name='content'"))
            if not result.fetchone():
                conn.execute(text("ALTER TABLE public_projects ADD COLUMN content TEXT"))
                conn.commit()
        
        print("  ✓ Database Initialization: OK")
    except Exception as e:
        print(f"  ⚠ Database Initialization FAILED: {e}")

    # 2. Redis Connection Check
    try:
        import redis
        r = redis.from_url(settings.REDIS_CONNECTION_URL, socket_timeout=5)
        r.ping()
        print("  ✓ Redis connection: OK")
    except Exception as e:
        print(f"  ⚠ Redis connection: FAILED - {e}")

    # 3. Qdrant Connection Check
    from app.modules.rag.vector_store import ensure_collection
    try:
        ensure_collection()
        print("  ✓ RAG Vector Store initialized")
    except Exception as e:
        print(f"  ⚠ RAG Initialization Warning: {e}")

    # 4. Start Background Schedulers
    try:
        from app.modules.github_trends.tasks import setup_github_trends_scheduler
        setup_github_trends_scheduler()
        print("  ✓ GitHub Trends Scheduler started")
    except Exception as e:
        print(f"  ⚠ Scheduler Initialization FAILED: {e}")

