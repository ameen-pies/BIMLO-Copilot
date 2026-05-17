"""
Centralized configuration for BIMLO Copilot Backend.
Uses pydantic-settings to load from environment variables / .env file.

Usage:
    from core.config import settings

    print(settings.neo4j_uri)
    print(settings.groq_api_key)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ──
    app_name: str = "BIMLO Copilot API"
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    data_dir: str = "/home/claude/bimlo-copilot/data"
    max_upload_mb: int = 50

    # ── LLM Providers ──
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    cf_api_key: str = ""
    cf_api_url: str = ""
    cf_backup_api_key: str = ""
    cf_backup_url: str = ""
    cf_backup2_api_key: str = ""
    cf_backup2_url: str = ""
    cf_news_api_key: str = ""
    cf_news_url: str = ""

    nvidia_api_key: str = ""
    openrouter_api_key: str = ""  # TEMP — testing only, remove after stress test
    elevenlabs_api_key: str = ""
    newsdata_api_key: str = ""
    firecrawl_api_key: str = ""

    # ── Neo4j ──
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # ── SMTP ──
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    smtp_from: str = ""
    contact_to: str = ""

    # ── Google OAuth ──
    google_client_id: str = ""

    # ── News ──
    news_cycle_days: int = 1
    news_cache_dir: str = ""
    news_page_size: int = 10
    news_max_age_days: int = 30

    # ── Vector Store ──
    chroma_host: str = "localhost"
    chroma_persist_dir: str = "./chroma_data"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Vision ──
    vision_model: str = "@cf/meta/llama-3.2-11b-vision-instruct"
    vision_max_dim: int = 2048
    skip_vision: bool = False

    # ── Reranker ──
    reranker_model: str = ""
    reranker_enabled: bool = True
    reranker_fetch_k: int = 30

    # ── Session / Token ──
    session_ttl_hours: int = 72
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 7

    # ── Override defaults with .env ──
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
