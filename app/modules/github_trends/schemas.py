from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class RepoBase(BaseModel):
    name: str
    owner: str
    full_name: str
    description: Optional[str]
    stars: int
    forks: int
    language: Optional[str]
    github_url: str
    avatar_url: Optional[str]

class RepoOut(RepoBase):
    id: int
    arabic_summary: Optional[str]
    storage_path: Optional[str]
    difficulty: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CategoryBase(BaseModel):
    name: str
    slug: str

class CategoryOut(CategoryBase):
    id: int

    class Config:
        from_attributes = True
