from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"

    mongodb_uri: str
    mongodb_db: str = "rag_db"
    mongodb_collection: str = "documents"
    mongodb_vector_index: str = "vector_index"

    rag_top_k: int = 5
    max_context_chunks: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120


settings = Settings()