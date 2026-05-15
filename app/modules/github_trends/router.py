from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from typing import List, Optional
from app.modules.github_trends import service

router = APIRouter()

@router.get("/repos")
async def get_trending_repos(
    language: Optional[str] = None,
    since: str = "daily"
):
    s = service.GitHubTrendsService()
    return await s.get_active_repos(language=language)

@router.get("/search")
async def search_repos(
    q: str
):
    s = service.GitHubTrendsService()
    return await s.semantic_search(q)

@router.get("/repos/{full_name:path}")
async def get_repo_details(full_name: str):
    s = service.GitHubTrendsService()
    repo = await s.get_repo_by_full_name(full_name)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo

@router.post("/refresh")
async def trigger_refresh(background_tasks: BackgroundTasks):
    """Manually trigger a refresh of trending repos."""
    async def run_refresh():
        s = service.GitHubTrendsService()
        repos = await s.fetch_trending_repos()
        await s.process_and_store_repos(repos)
        print(f"✅ Successfully processed {len(repos)} repositories")

    background_tasks.add_task(run_refresh)
    return {"message": "Refresh process started in background"}
