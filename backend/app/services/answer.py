import logging
import json
import re
import requests
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.openai_api_key)
lm_studio_client = OpenAI(
    api_key=settings.openai_api_key or "lm-studio",
    base_url=settings.lm_studio_base_url,
    timeout=600,
)


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    remaining_chars = settings.max_context_chars
    for c in chunks[: settings.max_context_chunks]:
        header = f"[source={c['source_name']} chunk={c['chunk_id']} score={c['score']:.4f}]\n"
        if remaining_chars <= len(header):
            break
        text = c["text"][: remaining_chars - len(header)]
        parts.append(f"{header}{text}")
        remaining_chars -= len(header) + len(text)
        if len(c["text"]) > len(text):
            break
    logger.debug(
        "RAG context: chunks=%d chars=%d limit=%d",
        len(parts),
        sum(len(part) for part in parts),
        settings.max_context_chars,
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


def _generate_with_lm_studio(prompt: str) -> str:
    response = lm_studio_client.chat.completions.create(
        model=settings.lm_studio_chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=settings.lm_studio_max_tokens,
    )
    raw_response = _dump_lm_studio_response(response)
    choices = getattr(response, "choices", None) or []
    logger.debug(
        "LM Studio response: model=%s choices=%d usage=%s raw=%s",
        getattr(response, "model", None),
        len(choices),
        getattr(response, "usage", None),
        raw_response,
    )

    if not choices:
        logger.warning("LM Studio zwróciło pustą listę choices. raw=%s", raw_response)
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = _message_content_to_text(getattr(message, "content", None) if message else None)
    reasoning_content = _message_content_to_text(
        getattr(message, "reasoning_content", None) if message else None
    )
    logger.debug(
        "LM Studio choice[0]: finish_reason=%s content_len=%d reasoning_len=%d content_preview=%r reasoning_preview=%r",
        getattr(choice, "finish_reason", None),
        len(content),
        len(reasoning_content),
        content[:500],
        reasoning_content[:500],
    )

    if not content:
        logger.warning(
            "LM Studio zwróciło pusty message.content. finish_reason=%s raw=%s",
            getattr(choice, "finish_reason", None),
            raw_response,
        )
    return content


def _dump_lm_studio_response(response: object) -> str:
    try:
        payload = response.model_dump(mode="json")
    except AttributeError:
        return repr(response)
    return json.dumps(payload, ensure_ascii=False, default=str)


def _message_content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


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
Pisz naturalnym językiem. Nie bądź zbyt zwięzły ale też nie przesadzaj z długością odpowiedzi.
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
    if settings.llm_backend == "lmstudio":
        return _clean_answer_text(_generate_with_lm_studio(prompt))
    else:
        return _clean_answer_text(_generate_with_openai(prompt))
