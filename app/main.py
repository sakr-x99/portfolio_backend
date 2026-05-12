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

from app.core.database import engine, Base
# Import all models to ensure they are registered with Base.metadata
from app.modules.public_portfolio import models as public_models
from app.modules.biz_management import models as biz_models

# Create tables
Base.metadata.create_all(bind=engine)

# Ensure 'gradient' column exists (manual migration for existing tables)
from sqlalchemy import text
with engine.connect() as conn:
    try:
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
    except Exception as e:
        print(f"Migration error: {e}")

@app.get("/")
def root():
    return {"message": "Welcome to the Portfolio API Modular Monolith!"}

from app.modules.public_portfolio.routers import router as public_router
from app.modules.biz_management.routers import router as biz_router
from app.modules.ai.routers import router as ai_router
from app.modules.rag.routers import router as rag_router

app.include_router(public_router, prefix=f"{settings.API_V1_STR}/public", tags=["Public Portfolio"])
app.include_router(biz_router, prefix=f"{settings.API_V1_STR}/admin", tags=["Business Management"])
app.include_router(ai_router, prefix=f"{settings.API_V1_STR}/ai", tags=["AI Integration"])
app.include_router(rag_router, prefix=f"{settings.API_V1_STR}/rag", tags=["RAG Pipeline"])

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    
    # Check Database
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection: OK")
    except Exception as e:
        print(f"⚠ Database connection: FAILED - {e}")

    # Check Redis
    try:
        import redis
        r = redis.from_url(settings.REDIS_CONNECTION_URL, socket_timeout=5)
        r.ping()
        print("✓ Redis connection: OK")
    except Exception as e:
        print(f"⚠ Redis connection: FAILED - {e}")

    # Ensure Qdrant collection is ready
    from app.modules.rag.vector_store import ensure_collection
    try:
        ensure_collection()
        print("✓ RAG Vector Store initialized")
    except Exception as e:
        print(f"⚠ RAG Initialization Warning: {e}")

