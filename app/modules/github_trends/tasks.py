from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.modules.github_trends.service import GitHubTrendsService
import logging

logger = logging.getLogger(__name__)

async def refresh_github_trends_task():
    logger.info("Starting scheduled GitHub Trends refresh...")
    try:
        service = GitHubTrendsService()
        repos = await service.fetch_trending_repos(since="daily")
        await service.process_and_store_repos(repos)
        logger.info(f"Successfully refreshed {len(repos)} trending repositories.")
    except Exception as e:
        logger.error(f"Error during GitHub Trends refresh task: {str(e)}")

def setup_github_trends_scheduler():
    scheduler = AsyncIOScheduler()
    # Run every 24 hours
    scheduler.add_job(refresh_github_trends_task, 'interval', hours=24)
    scheduler.start()
    return scheduler
