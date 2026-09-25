from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "Indian Pharma Source Lookup"
    app_version: str = "0.1.0"
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_service_role_key: str = ""
    supabase_db_url: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
