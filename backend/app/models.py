from datetime import datetime, timezone
from typing import Any


def make_chunk_doc(
    *,
    source_id: str,
    source_name: str,
    chunk_id: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_name": source_name,
        "chunk_id": chunk_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc),
    }