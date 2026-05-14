import httpx
import bs4
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.supabase_client import supabase
from app.modules.github_trends.models import TrendingRepo, TrendingCategory
from app.modules.ai.service import AIService # Assuming this exists
import asyncio

GITHUB_TRENDING_URL = "https://github.com/trending"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

class GitHubTrendsService:
    def __init__(self, db: Session):
        self.db = db
        self.ai_service = AIService()

    async def fetch_trending_repos(self, since: str = "daily", language: str = None) -> List[Dict[str, Any]]:
        url = GITHUB_TRENDING_URL
        if language:
            url += f"/{language}"
        url += f"?since={since}"

        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return self._parse_github_html(response.text)

    def _parse_github_html(self, html: str) -> List[Dict[str, Any]]:
        soup = bs4.BeautifulSoup(html, "lxml")
        repos = []
        for article in soup.select("article.Box-row"):
            # Repo Info
            link_tag = article.select_one("h2.h3 a")
            if not link_tag: continue
            
            href = link_tag["href"].strip("/")
            owner, name = href.split("/")
            full_name = f"{owner}/{name}"
            
            # Description
            desc_tag = article.select_one("p.col-9")
            description = desc_tag.get_text(strip=True) if desc_tag else ""
            
            # Stats
            stars_tag = article.select_one("a[href$='/stargazers']")
            stars = int(stars_tag.get_text(strip=True).replace(",", "")) if stars_tag else 0
            
            forks_tag = article.select_one("a[href$='/forks']")
            forks = int(forks_tag.get_text(strip=True).replace(",", "")) if forks_tag else 0
            
            # Language
            lang_tag = article.select_one("span[itemprop='programmingLanguage']")
            language = lang_tag.get_text(strip=True) if lang_tag else None
            
            # Avatar
            avatar_tag = article.select_one("img.avatar")
            avatar_url = avatar_tag["src"] if avatar_tag else None

            repos.append({
                "name": name,
                "owner": owner,
                "full_name": full_name,
                "description": description,
                "stars": stars,
                "forks": forks,
                "language": language,
                "github_url": f"https://github.com/{full_name}",
                "avatar_url": avatar_url
            })
        return repos

    async def process_and_store_repos(self, repos: List[Dict[str, Any]]):
        for repo_data in repos:
            # Check if exists
            db_repo = self.db.query(TrendingRepo).filter(TrendingRepo.full_name == repo_data["full_name"]).first()
            
            if not db_repo:
                db_repo = TrendingRepo(**repo_data)
                self.db.add(db_repo)
                self.db.commit()
                self.db.refresh(db_repo)
                
                # Generate AI Content
                await self.generate_and_store_ai_content(db_repo)
            else:
                # Update stats
                db_repo.stars = repo_data["stars"]
                db_repo.forks = repo_data["forks"]
                self.db.commit()

    async def generate_and_store_ai_content(self, repo: TrendingRepo):
        prompt = f"""
        Analyze this GitHub repository:
        Name: {repo.full_name}
        Description: {repo.description}
        Language: {repo.language}
        
        Generate a premium Arabic markdown explanation. 
        Focus on:
        - What it does (concise)
        - Why use it
        - Real-world use cases
        - Strengths & Weaknesses
        - Difficulty level
        - Best scenarios
        - Alternatives
        
        Use modern Arabic technical writing. Premium developer tone.
        Include frontmatter with: title, slug, language, category, stars, difficulty, topics.
        """
        
        # Use ai_manager
        from app.services.ai.manager import ai_manager
        messages = [{"role": "user", "content": prompt}]
        content = await ai_manager.generate(messages=messages)
        
        # Store in Supabase
        filename = f"{repo.full_name.replace('/', '-')}.md"
        path = f"repos/{filename}"
        
        if supabase:
            try:
                # Store content in Supabase
                supabase.storage.from_("github-trends").upload(
                    path=path,
                    file=content.encode("utf-8"),
                    file_options={"upsert": "true", "content-type": "text/markdown"}
                )
                
                repo.storage_path = path
                
                # Simple extraction for summary (first 200 chars after frontmatter)
                repo.arabic_summary = content.split("---")[-1].strip()[:200] + "..."
                self.db.commit()
                
                # Index in Qdrant
                await self.index_repo_in_qdrant(repo, content)
            except Exception as e:
                print(f"Error storing/indexing: {e}")

    async def index_repo_in_qdrant(self, repo: TrendingRepo, content: str):
        from app.modules.rag.embeddings import embed_texts
        from app.modules.rag.vector_store import _get_client
        from app.modules.rag import config as rag_config
        from qdrant_client.models import PointStruct
        import uuid

        # Clean content for embedding
        clean_text = content.split("---")[-1].strip()
        embedding = embed_texts([clean_text])[0]

        client = _get_client()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": clean_text,
                "full_name": repo.full_name,
                "source": "github_trends",
                "language": repo.language,
                "stars": repo.stars
            }
        )
        
        client.upsert(
            collection_name=rag_config.QDRANT_COLLECTION,
            points=[point]
        )

    async def semantic_search(self, query: str, limit: int = 10) -> List[TrendingRepo]:
        from app.modules.rag.embeddings import embed_query
        from app.modules.rag.vector_store import _get_client
        from app.modules.rag import config as rag_config
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_vector = embed_query(query)
        client = _get_client()
        
        results = client.search(
            collection_name=rag_config.QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="github_trends"))]
            ),
            limit=limit
        )

        repo_names = [hit.payload["full_name"] for hit in results]
        return self.db.query(TrendingRepo).filter(TrendingRepo.full_name.in_(repo_names)).all()
