import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    sarvam_api_key: str = Field(default="", env="SARVAM_API_KEY")
    sarvam_stt_url: str = Field(default="https://api.sarvam.ai/speech-to-text", env="SARVAM_STT_URL")
    
    # Hugging Face Token for fast authenticated downloads
    hf_token: str = Field(default="", env="HF_TOKEN")
    
    # LLM Settings
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL")
    
    # Vector DB & Guardrail Defaults
    chroma_path: str = Field(default="./data/chroma", env="CHROMA_PATH")
    metrics_db_path: str = Field(default="./data/metrics.db", env="METRICS_DB_PATH")
    similarity_threshold: float = Field(default=0.25, env="SIMILARITY_THRESHOLD")
    default_top_k: int = Field(default=3, env="DEFAULT_TOP_K")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
