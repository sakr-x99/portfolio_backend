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
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.DATABASE_URL:
            # Handle postgres:// vs postgresql://
            return self.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_SERVER}/{self.DB_NAME}"
    
    # JWT
    SECRET_KEY: str = "YOUR_SUPER_SECRET_KEY_HERE_CHANGE_IN_PROD"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    
    # Cache (Renamed to bypass auto-detection)
    CACHE_URL: str = "redis://localhost:6379"

    # AI Provider System
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    PRIMARY_AI_PROVIDER: str = "groq"
    FALLBACK_PROVIDER: str = "gemini"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # RAG / Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "portfolio_knowledge"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"

settings = Settings()
