from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.modules.github_trends import models, schemas, service

router = APIRouter()

@router.get("/repos", response_model=List[schemas.RepoOut])
async def get_trending_repos(
    language: Optional[str] = None,
    category: Optional[str] = None,
    since: str = "daily",
    db: Session = Depends(get_db)
):
    query = db.query(models.TrendingRepo).filter(models.TrendingRepo.is_active == True)
    
    if language:
        query = query.filter(models.TrendingRepo.language == language)
    
    return query.order_by(models.TrendingRepo.stars.desc()).all()

@router.get("/search", response_model=List[schemas.RepoOut])
async def search_repos(
    q: str,
    db: Session = Depends(get_db)
):
    s = service.GitHubTrendsService(db)
    return await s.semantic_search(q)

@router.get("/repos/{full_name:path}", response_model=schemas.RepoOut)
async def get_repo_details(full_name: str, db: Session = Depends(get_db)):
    repo = db.query(models.TrendingRepo).filter(models.TrendingRepo.full_name == full_name).first()
    if not repo:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo

@router.post("/refresh")
async def trigger_refresh(db: Session = Depends(get_db)):
    """Manually trigger a refresh of trending repos."""
    s = service.GitHubTrendsService(db)
    repos = await s.fetch_trending_repos()
    await s.process_and_store_repos(repos)
    return {"message": f"Successfully processed {len(repos)} repositories"}
