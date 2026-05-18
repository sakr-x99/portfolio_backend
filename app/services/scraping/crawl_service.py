import asyncio
from typing import Optional, Dict, Any

class CrawlService:
    def __init__(self):
        self.crawler = None

    async def get_crawler(self):
        if self.crawler is None:
            from crawl4ai import AsyncWebCrawler
            self.crawler = AsyncWebCrawler(verbose=True)
            await self.crawler.__aenter__()
        return self.crawler

    async def crawl_url(self, url: str) -> Optional[str]:
        """
        Crawls a URL and returns the markdown content.
        """
        try:
            crawler = await self.get_crawler()
            result = await crawler.arun(url=url)
            return result.markdown
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None

    async def close(self):
        if self.crawler:
            await self.crawler.__aexit__(None, None, None)
            self.crawler = None

# Global instance
crawl_service = CrawlService()
