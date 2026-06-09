import logging
from traceback import print_stack

import requests
from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)


def _embed_with_openai(texts: list[str]) -> list[list[float]]:
    try:
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        return [e.embedding for e in response.data]
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        print_stack()
        raise e


def _embed_with_ollama(texts: list[str]) -> list[list[float]]:
    url = f"{settings.ollama_base_url}/api/embed"
    embeddings = []
    for text in texts:
        payload = {"model": settings.ollama_embedding_model, "input": text}
        logger.debug("Ollama embed: model=%s", settings.ollama_embedding_model)
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        embeddings.append(data["embeddings"][0])
    return embeddings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def embed_texts(texts: list[str]) -> list[list[float]]:
    logger.info("Embedding backend: %s", settings.embedding_backend)
    if settings.embedding_backend == "ollama":
        return _embed_with_ollama(texts)
    return _embed_with_openai(texts)
