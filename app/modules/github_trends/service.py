import httpx
import bs4
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from app.core.config import settings
from app.core.supabase_client import supabase
from app.core.mongodb_client import mongodb_client
import asyncio
from app.services.ai.manager import github_trends_ai
from app.services.scraping.crawl_service import crawl_service

GITHUB_TRENDING_URL = "https://github.com/trending"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

class GitHubTrendsService:
    def __init__(self):
        self.db = mongodb_client.get_db()
        self.collection = self.db["trending_repos"]

    async def fetch_trending_repos(self, since: str = "daily", language: str = None) -> List[Dict[str, Any]]:
        url = GITHUB_TRENDING_URL
        if language:
            url += f"/{language}"
        url += f"?since={since}"

        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            repos = self._parse_github_html(response.text)
            # Tag repos with the since period
            for repo in repos:
                repo["since"] = since
            return repos

    def _parse_github_html(self, html: str) -> List[Dict[str, Any]]:
        soup = bs4.BeautifulSoup(html, "lxml")
        repos = []
        for i, article in enumerate(soup.select("article.Box-row")):
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
                "rank": i + 1,
                "name": name,
                "owner": owner,
                "full_name": full_name,
                "description": description,
                "stars": stars,
                "forks": forks,
                "language": language,
                "github_url": f"https://github.com/{full_name}",
                "avatar_url": avatar_url,
                "is_active": True,
                "updated_at": datetime.utcnow()
            })
        return repos

    async def fetch_readme(self, full_name: str) -> Optional[str]:
        """Fetch raw README.md content from GitHub API."""
        url = f"https://api.github.com/repos/{full_name}/readme"
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github.v3+json"},
            timeout=15.0
        ) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                import base64
                return base64.b64decode(data["content"].replace("\n", "")).decode("utf-8")
        return None

    async def process_and_store_repos(self, repos: List[Dict[str, Any]]):
        # Set all as inactive first to refresh the top list
        await self.collection.update_many({}, {"$set": {"is_active": False}})
        
        for repo_data in repos:
            # Check if exists in MongoDB
            existing_repo = await self.collection.find_one({"full_name": repo_data["full_name"]})
            
            if not existing_repo:
                repo_data["created_at"] = datetime.utcnow()
                # Insert but keep inactive until we have content
                repo_data["is_active"] = False 
                result = await self.collection.insert_one(repo_data)
                repo_data["_id"] = result.inserted_id
                
                # Fetch and store raw README
                readme_content = await self.fetch_readme(repo_data["full_name"])
                if readme_content:
                    await self.collection.update_one(
                        {"_id": repo_data["_id"]},
                        {"$set": {"readme_content": readme_content, "is_active": True}}
                    )
                else:
                    # Fallback: Activate anyway so it shows up (maybe add error flag later)
                    await self.collection.update_one(
                        {"_id": repo_data["_id"]},
                        {"$set": {"is_active": True}}
                    )
                
                # Generate AI Content (Best effort, don't block activation)
                try:
                    await self.generate_and_store_ai_content(repo_data)
                except Exception as e:
                    print(f"AI Generation failed for {repo_data['full_name']}: {e}")
            else:
                # Update stats and rank
                update_data = {
                    "stars": repo_data["stars"],
                    "forks": repo_data["forks"],
                    "rank": repo_data["rank"],
                    "is_active": True, # Reactivate immediately if it already has AI content
                    "updated_at": datetime.utcnow()
                }
                
                # Fetch updated README if missing
                if not existing_repo.get("readme_content"):
                    readme = await self.fetch_readme(repo_data["full_name"])
                    if readme:
                        update_data["readme_content"] = readme
                
                # Only re-generate AI content if it doesn't exist or is older than 24h
                needs_ai = False
                if "arabic_summary" not in existing_repo:
                    needs_ai = True
                    update_data["is_active"] = False # Hide until AI is done
                else:
                    updated_at = existing_repo.get("updated_at")
                    if updated_at and (datetime.utcnow() - updated_at).total_seconds() > 86400:
                          needs_ai = True
                
                await self.collection.update_one(
                    {"_id": existing_repo["_id"]},
                    {"$set": update_data}
                )
                
                if needs_ai:
                    repo_data["_id"] = existing_repo["_id"]
                    await self.generate_and_store_ai_content(repo_data)
                    # Activate after AI
                    await self.collection.update_one({"_id": repo_data["_id"]}, {"$set": {"is_active": True}})

            # Small delay to be "one by one" and avoid rate limits
            await asyncio.sleep(2)

    async def generate_and_store_ai_content(self, repo: Dict[str, Any]):
        try:
            # Use crawl4ai to get full repo content if possible
            try:
                repo_content = await crawl_service.crawl_url(repo["github_url"])
            except Exception:
                repo_content = None
            
            prompt = f"""
            Analyze this GitHub repository:
            Name: {repo['full_name']}
            Description: {repo['description']}
            Language: {repo['language']}
            
            Full Repository Content (Extracted via Crawl4AI):
            {repo_content if repo_content else "No additional content found."}
            
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
            
            # Use github_trends_ai
            messages = [{"role": "user", "content": prompt}]
            content = await github_trends_ai.generate(messages=messages)
            
            # Store in Supabase (Optional)
            if supabase:
                try:
                    filename = f"{repo['full_name'].replace('/', '-')}.md"
                    path = f"repos/{filename}"
                    
                    supabase.storage.from_("github-trends").upload(
                        path=path,
                        file=content.encode("utf-8"),
                        file_options={"upsert": "true", "content-type": "text/markdown"}
                    )
                    
                    arabic_summary = content.split("---")[-1].strip()[:500]
                    public_url = supabase.storage.from_("github-trends").get_public_url(path)
                    
                    await self.collection.update_one(
                        {"_id": repo["_id"]},
                        {"$set": {
                            "storage_path": path,
                            "analysis_url": public_url,
                            "arabic_summary": arabic_summary,
                            "updated_at": datetime.utcnow()
                        }}
                    )
                    
                    # Index in Qdrant
                    await self.index_repo_in_qdrant(repo, content)
                except Exception as e:
                    print(f"Supabase/Qdrant storage skipped: {e}")
        except Exception as e:
            print(f"AI Generation skipped for {repo['full_name']}: {e}")

    async def index_repo_in_qdrant(self, repo: Dict[str, Any], content: str):
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
                "full_name": repo["full_name"],
                "source": "github_trends",
                "language": repo["language"],
                "stars": repo["stars"]
            }
        )
        
        client.upsert(
            collection_name=rag_config.QDRANT_COLLECTION,
            points=[point]
        )

    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
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
        
        # Fetch from MongoDB
        cursor = self.collection.find({"full_name": {"$in": repo_names}})
        repos = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string for JSON serialization
        for r in repos:
            r["_id"] = str(r["_id"])
            
        return repos

    async def get_active_repos(self, language: Optional[str] = None, since: str = "daily", limit: int = 25) -> List[Dict[str, Any]]:
        query = {"is_active": True, "since": since}
        if language:
            query["language"] = language
            
        cursor = self.collection.find(query).sort("rank", 1).limit(limit)
        repos = await cursor.to_list(length=limit)
        
        for r in repos:
            r["_id"] = str(r["_id"])
        return repos

    async def get_available_languages(self) -> List[str]:
        """Get list of all unique languages from active repos."""
        languages = await self.collection.distinct("language", {"is_active": True, "language": {"$ne": None}})
        return sorted([lang for lang in languages if lang])

    async def process_and_store_repos(self, repos: List[Dict[str, Any]], since: str = "daily"):
        # Mark repos from this period as inactive first
        await self.collection.update_many({"since": since}, {"$set": {"is_active": False}})
        
        for repo_data in repos:
            repo_data["since"] = since
            existing_repo = await self.collection.find_one({"full_name": repo_data["full_name"]})
            
            if not existing_repo:
                repo_data["created_at"] = datetime.utcnow()
                repo_data["is_active"] = False 
                result = await self.collection.insert_one(repo_data)
                repo_data["_id"] = result.inserted_id
                
                readme_content = await self.fetch_readme(repo_data["full_name"])
                if readme_content:
                    await self.collection.update_one(
                        {"_id": repo_data["_id"]},
                        {"$set": {"readme_content": readme_content, "is_active": True}}
                    )
                else:
                    await self.collection.update_one(
                        {"_id": repo_data["_id"]},
                        {"$set": {"is_active": True}}
                    )
                
                try:
                    await self.generate_and_store_ai_content(repo_data)
                except Exception as e:
                    print(f"AI Generation failed for {repo_data['full_name']}: {e}")
            else:
                update_data = {
                    "stars": repo_data["stars"],
                    "forks": repo_data["forks"],
                    "rank": repo_data["rank"],
                    "is_active": True,
                    "updated_at": datetime.utcnow(),
                    "since": since
                }
                
                if not existing_repo.get("readme_content"):
                    readme = await self.fetch_readme(repo_data["full_name"])
                    if readme:
                        update_data["readme_content"] = readme
                
                needs_ai = False
                if "arabic_summary" not in existing_repo:
                    needs_ai = True
                    update_data["is_active"] = False
                else:
                    updated_at = existing_repo.get("updated_at")
                    if updated_at and (datetime.utcnow() - updated_at).total_seconds() > 86400:
                          needs_ai = True
                
                await self.collection.update_one(
                    {"_id": existing_repo["_id"]},
                    {"$set": update_data}
                )
                
                if needs_ai:
                    repo_data["_id"] = existing_repo["_id"]
                    await self.generate_and_store_ai_content(repo_data)
                    await self.collection.update_one({"_id": repo_data["_id"]}, {"$set": {"is_active": True}})

            await asyncio.sleep(2)

    async def get_repo_by_full_name(self, full_name: str) -> Optional[Dict[str, Any]]:
        repo = await self.collection.find_one({"full_name": full_name})
        if repo:
            repo["_id"] = str(repo["_id"])
            # If no stored README, fetch it live from GitHub API
            if not repo.get("readme_content"):
                readme = await self.fetch_readme(full_name)
                if readme:
                    repo["readme_content"] = readme
                    # Store it for future requests
                    await self.collection.update_one(
                        {"full_name": full_name},
                        {"$set": {"readme_content": readme}}
                    )
        return repo
