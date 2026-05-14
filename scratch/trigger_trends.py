import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.getcwd())

from app.core.database import SessionLocal
from app.modules.github_trends.service import GitHubTrendsService

async def main():
    print("🚀 Manually triggering GitHub Trends refresh...")
    db = SessionLocal()
    try:
        service = GitHubTrendsService(db)
        print("  → Fetching trending repos from GitHub...")
        repos = await service.fetch_trending_repos(since="daily")
        print(f"  ✓ Found {len(repos)} repos. Processing and generating AI content...")
        await service.process_and_store_repos(repos)
        print("  ✅ Done! Database populated.")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
