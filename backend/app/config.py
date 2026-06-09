from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"

    # Tryb LLM: "openai", "ollama" lub "lmstudio"
    llm_backend: str = "ollama"
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_chat_model: str = "google/gemma-4-12B"
    lm_studio_max_tokens: int = 1024
    ollama_chat_model: str = "doctor"
    ollama_base_url: str = "http://localhost:11434"

    # Tryb embeddingów: "openai" lub "ollama"
    embedding_backend: str = "ollama"
    ollama_embedding_model: str = "bge-m3"

    mongodb_uri: str
    mongodb_db: str = "rag_db"
    mongodb_collection: str = "documents"
    mongodb_vector_index: str = "vector_index"

    rag_top_k: int = 5
    rag_similarity_device: str = "auto"
    rag_min_context_score: float = 0.2
    rag_always_keep_top_chunk: bool = True
    max_context_chunks: int = 5
    max_context_chars: int = 4000
    chunk_size: int = 800
    chunk_overlap: int = 120

    log_level: str = "DEBUG"


settings = Settings()
