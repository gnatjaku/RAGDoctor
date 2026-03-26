from traceback import print_stack

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def embed_texts(texts: list[str]) -> list[list[float]]:

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

