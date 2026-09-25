from functools import lru_cache

from supabase import Client, create_client

from app.core.config import get_settings


@lru_cache
def get_supabase_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@lru_cache
def get_readonly_supabase_client() -> Client:
    settings = get_settings()
    key = settings.supabase_publishable_key or settings.supabase_service_role_key
    if not settings.supabase_url or not key:
        raise RuntimeError("SUPABASE_URL and a Supabase API key are required")
    return create_client(settings.supabase_url, key)
