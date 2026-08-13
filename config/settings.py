import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    elevenlabs_api_key: str = Field(default="", env="ELEVENLABS_API_KEY")
    sarvam_api_key: str = Field(default="", env="SARVAM_API_KEY")
    sarvam_stt_url: str = Field(default="https://api.sarvam.ai/speech-to-text", env="SARVAM_STT_URL")
    hf_token: str = Field(default="", env="HF_TOKEN")
    
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL")
    
    chroma_path: str = Field(default="./data/chroma", env="CHROMA_PATH")
    metrics_db_path: str = Field(default="./data/metrics.db", env="METRICS_DB_PATH")
    
    # Calibrated threshold for 5-6 out-of-domain refusals
    similarity_threshold: float = Field(default=0.22, env="SIMILARITY_THRESHOLD")
    default_top_k: int = Field(default=3, env="DEFAULT_TOP_K")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
