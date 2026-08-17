import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    elevenlabs_api_key: str = Field(default="", validation_alias="ELEVENLABS_API_KEY")
    sarvam_api_key: str = Field(default="", validation_alias="SARVAM_API_KEY")
    sarvam_stt_url: str = Field(default="https://api.sarvam.ai/speech-to-text", validation_alias="SARVAM_STT_URL")
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")
    
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="groq/compound-mini", validation_alias="GROQ_MODEL")
    
    chroma_path: str = Field(default="./data/chroma", validation_alias="CHROMA_PATH")
    metrics_db_path: str = Field(default="./data/metrics.db", validation_alias="METRICS_DB_PATH")
    
    # Calibrated threshold for zero false-negative refusals across MSMARCO-XI corpus
    similarity_threshold: float = Field(default=0.15, validation_alias="SIMILARITY_THRESHOLD")
    default_top_k: int = Field(default=3, validation_alias="DEFAULT_TOP_K")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
