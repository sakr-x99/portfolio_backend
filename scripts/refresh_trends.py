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
        
        # Fetch for all time periods
        periods = ["daily", "weekly", "monthly"]
        total_repos = 0
        
        for since in periods:
            print(f"\n Fetching {since} trending repos from GitHub...")
            repos = await service.fetch_trending_repos(since=since)
            print(f"✅ Found {len(repos)} trending repos for {since}.")
            
            if repos:
                print(f"💾 Processing and storing {since} repos in MongoDB...")
                await service.process_and_store_repos(repos)
                total_repos += len(repos)
        
        print(f"\n🎉 Refresh complete! {total_repos} total repos processed across all periods.")
        
    except Exception as e:
        print(f"❌ Error during refresh: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
