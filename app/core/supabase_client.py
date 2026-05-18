from app.core.config import settings

_supabase_client = None
_initialized = False

def get_supabase_client():
    global _supabase_client, _initialized
    if not _initialized:
        _initialized = True
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            _supabase_client = None
        else:
            try:
                from supabase import create_client
                _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            except Exception as e:
                print(f"  ⚠ Supabase Initialization FAILED: {e}")
                _supabase_client = None
    return _supabase_client

class _LazySupabase:
    """Proxy that defers supabase import until first access."""
    def __getattr__(self, name):
        client = get_supabase_client()
        if client is None:
            return None
        return getattr(client, name)
    
    def __bool__(self):
        return get_supabase_client() is not None

# Singleton instance for general use
supabase = _LazySupabase()
