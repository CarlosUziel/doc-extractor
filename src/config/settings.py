from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = Field(
        default=None, description="API key for OpenAI services."
    )
    GEMINI_API_KEY: str | None = Field(
        default=None, description="API key for Gemini services."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
