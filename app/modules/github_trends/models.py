from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.core.database import Base
from datetime import datetime

class TrendingCategory(Base):
    __tablename__ = "trending_categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True) # e.g., "AI", "Backend"
    slug = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    repos = relationship("TrendingRepo", back_populates="category")

class TrendingRepo(Base):
    __tablename__ = "trending_repos"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    owner = Column(String, index=True)
    full_name = Column(String, unique=True, index=True) # owner/name
    description = Column(Text, nullable=True)
    stars = Column(Integer, default=0)
    forks = Column(Integer, default=0)
    language = Column(String, index=True, nullable=True)
    topics = Column(JSON, default=[])
    github_url = Column(String)
    avatar_url = Column(String, nullable=True)
    
    # AI Content
    arabic_summary = Column(Text, nullable=True)
    storage_path = Column(String, nullable=True) # path in Supabase
    difficulty = Column(String, nullable=True) # Beginner, Intermediate, Advanced
    
    category_id = Column(Integer, ForeignKey("trending_categories.id"), nullable=True)
    category = relationship("TrendingCategory", back_populates="repos")
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProjectBookmark(Base):
    __tablename__ = "project_bookmarks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True) # Assuming auth system exists
    repo_id = Column(Integer, ForeignKey("trending_repos.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
