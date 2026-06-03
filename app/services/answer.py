import logging
import re
import requests
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for c in chunks[: settings.max_context_chunks]:
        parts.append(
            f"[source={c['source_name']} chunk={c['chunk_id']} score={c['score']:.4f}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def _generate_with_openai(prompt: str) -> str:
    response = client.responses.create(
        model=settings.openai_chat_model,
        input=prompt,
    )
    return response.output_text


def _generate_with_ollama(prompt: str) -> str:
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_chat_model,
        "prompt": prompt,
        "stream": False,
    }
    logger.debug("Ollama URL: %s", url)
    logger.debug("Ollama model: %s", settings.ollama_chat_model)
    logger.debug("Prompt (pierwsze 200 znaków): %.200s", prompt)
    response = requests.post(url, json=payload, timeout=600)
    logger.debug("Ollama HTTP status: %s", response.status_code)
    response.raise_for_status()
    result = response.json().get("response", "")
    logger.debug("Ollama odpowiedź (pierwsze 200 znaków): %.200s", result)
    return result


_SOURCE_TRAIL_RE = re.compile(
    r"\s*\(source:\s*[^()]*?(?:chunk=[^()]*?)?\)\s*",
    flags=re.IGNORECASE,
)


def _clean_answer_text(answer: str) -> str:
    cleaned = _SOURCE_TRAIL_RE.sub("", answer)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def generate_grounded_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    prompt = f"""Odpowiadasz wyłącznie na podstawie poniższego kontekstu medycznego.
Jeśli odpowiedź nie wynika z kontekstu, powiedz że nie wiesz.
Bądź zwięzły i pisz naturalnym językiem.
Nie wklejaj do odpowiedzi technicznych adnotacji typu source, chunk, score ani nawiasów cytujących.
Jeśli chcesz oprzeć odpowiedź na źródłach, zrób to niejawnie w treści, a źródła będą pokazane osobno w interfejsie.

Kontekst:
{context}

Pytanie:
{question}
"""

    logger.info("Backend LLM: %s", settings.llm_backend)

    if settings.llm_backend == "ollama":
        return _clean_answer_text(_generate_with_ollama(prompt))
    else:
        return _clean_answer_text(_generate_with_openai(prompt))
