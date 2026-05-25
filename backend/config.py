# backend/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_base_url: str = "https://api.minimax.chat/v1"
    anthropic_api_key: str = ""


settings = Settings()
