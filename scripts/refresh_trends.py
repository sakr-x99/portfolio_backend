import os
import sys
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modules.github_trends.service import GitHubTrendsService

async def main():
    print("🚀 Starting GitHub Trends Refresh via GitHub Action...")
    
    try:
        service = GitHubTrendsService()
        
        # 1. Fetch Trending Repos
        print(" Fetching trending repos from GitHub...")
        repos = await service.fetch_trending_repos(since="daily")
        print(f"✅ Found {len(repos)} trending repos.")
        
        if not repos:
            print("⚠️ No repos found. Exiting.")
            return

        # 2. Process and Store (README + AI Content)
        print("💾 Processing and storing in MongoDB...")
        await service.process_and_store_repos(repos)
        print("🎉 Refresh complete! Database updated.")
        
    except Exception as e:
        print(f"❌ Error during refresh: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
