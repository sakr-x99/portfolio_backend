from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Portfolio API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Database (Renamed to bypass auto-detection)
    DB_SERVER: str = "localhost"
    DB_USER: str = "admin"
    DB_PASSWORD: str = "adminpassword"
    DB_NAME: str = "portfolio"
    DATABASE_URL: str | None = None
    MONGODB_URL: str | None = None
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.DATABASE_URL:
            # Handle postgres:// vs postgresql://
            return self.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_SERVER}/{self.DB_NAME}"
    
    # JWT
    SECRET_KEY: str = "YOUR_SUPER_SECRET_KEY_HERE_CHANGE_IN_PROD"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    
    # Cache (Supports both REDIS_URL and CACHE_URL for cloud compatibility)
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_URL: str | None = None

    @property
    def REDIS_CONNECTION_URL(self) -> str:
        # Priority: REDIS_URL > CACHE_URL > default
        url = self.REDIS_URL or self.CACHE_URL or "redis://localhost:6379"
        # Upstash rediss:// handling - redis-py handles this automatically if URL is correct
        return url

    # AI Provider System 1 (Main Agent - Sakr AI)
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    PRIMARY_AI_PROVIDER: str = "groq"
    FALLBACK_PROVIDER: str = "gemini"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # AI Provider System 2 (Secondary Agent - GitHub Trends)
    GROQ_API_KEY_2: str | None = None
    GEMINI_API_KEY_2: str | None = None
    PRIMARY_AI_PROVIDER_2: str = "groq"
    FALLBACK_PROVIDER_2: str = "gemini"
    GROQ_MODEL_2: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL_2: str = "gemini-2.5-flash"

    # RAG / Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "portfolio_knowledge"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Supabase
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    SUPABASE_BUCKET: str = "portfolio-assets"

    class Config:
        env_file = ".env"

settings = Settings()
