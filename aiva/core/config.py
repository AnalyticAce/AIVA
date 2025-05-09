from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4-turbo"
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    environment: str = "development"
    
    api_key_header: str = "X-API-KEY"
    api_key: Optional[str] = None
    rate_limit_requests: int = 60
    rate_limit_period: int = 60
    cors_origins: List[str] = ["*"]
    
    log_level: str = "INFO"
    enable_sentry: bool = False
    sentry_dsn: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()