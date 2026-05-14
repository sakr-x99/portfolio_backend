from supabase import create_client, Client
from app.core.config import settings

def get_supabase_client() -> Client | None:
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        return None
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Singleton instance for general use
supabase: Client | None = get_supabase_client()
