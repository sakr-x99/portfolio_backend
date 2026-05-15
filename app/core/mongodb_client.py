from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class MongoDBClient:
    def __init__(self):
        self.client = None
        self.db = None

    def connect(self):
        if settings.MONGODB_URL:
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            # Extract DB name from URL or use default
            db_name = settings.DB_NAME or "portfolio"
            self.db = self.client[db_name]
        else:
            print("⚠ MONGODB_URL not set. MongoDB operations will fail.")

    def get_db(self):
        if self.db is None:
            self.connect()
        return self.db

mongodb_client = MongoDBClient()
