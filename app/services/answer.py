from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for c in chunks[: settings.max_context_chunks]:
        parts.append(
            f"[source={c['source_name']} chunk={c['chunk_id']} score={c['score']:.4f}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def generate_grounded_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    prompt = f"""You answer questions using only the retrieved context.
If the answer is not supported by the context, say you don't know.
Be concise and include source names in the answer.

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model=settings.openai_chat_model,
        input=prompt,
    )
    return response.output_text