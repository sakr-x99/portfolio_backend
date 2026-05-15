from supabase import create_client, Client
from app.core.config import settings

def get_supabase_client() -> Client | None:
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        return None
    try:
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    except Exception as e:
        print(f"  ⚠ Supabase Initialization FAILED: {e}")
        return None

# Singleton instance for general use
supabase: Client | None = get_supabase_client()
