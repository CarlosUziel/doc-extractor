"""Configuration settings for the application."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Manages application settings using Pydantic.

    Settings are loaded from a .env file and can be overridden by environment
    variables. This class defines the expected environment variables and their types.

    Attributes:
        openai_api_key: API key for OpenAI services.
        gemini_api_key: API key for Gemini services.
    """

    openai_api_key: str | None = Field(
        default=None, description="API key for OpenAI services."
    )
    gemini_api_key: str | None = Field(
        default=None, description="API key for Gemini services."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
