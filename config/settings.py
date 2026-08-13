from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    sarvam_api_key: str = Field(default="demo_sarvam_key", env="SARVAM_API_KEY")
    sarvam_stt_url: str = Field(default="https://api.sarvam.ai/speech-to-text", env="SARVAM_STT_URL")
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")
    ollama_model: str = Field(default="llama3.2:1b", env="OLLAMA_MODEL")
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    chroma_path: str = Field(default="./data/chroma", env="CHROMA_PATH")
    metrics_db_path: str = Field(default="./data/metrics.db", env="METRICS_DB_PATH")
    similarity_threshold: float = Field(default=0.65, env="SIMILARITY_THRESHOLD")
    default_top_k: int = Field(default=3, env="DEFAULT_TOP_K")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
