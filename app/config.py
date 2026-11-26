from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@127.0.0.1:5432/rag"
    embedding_model: str = "intfloat/multilingual-e5-base"
    claude_model: str = "claude-opus-5-5"
    hybrid_lexical_weight: float = 0.0
    rrf_k: int = 60
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # empty string disables reranking
    rerank_depth: int = Field(default=50, ge=1, le=50)  # 50 = retrieval.DEPTH
    anthropic_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
