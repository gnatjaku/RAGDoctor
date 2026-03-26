import math

from app.db import collection
from app.config import settings
from app.services.embeddings import embed_texts


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_chunks(question: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.rag_top_k
    query_vector = embed_texts([question])[0]

    # Brute-force KNN – działa z lokalnym MongoDB (bez Atlas)
    docs = list(collection.find(
        {"embedding": {"$exists": True}},
        {"_id": 0, "source_name": 1, "chunk_id": 1, "text": 1, "metadata": 1, "embedding": 1},
    ))

    scored = []
    for doc in docs:
        embedding = doc.pop("embedding")
        score = _cosine_similarity(query_vector, embedding)
        doc["score"] = score
        scored.append(doc)

    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]
