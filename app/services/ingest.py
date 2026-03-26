from pymongo import UpdateOne
from app.db import collection
from app.models import make_chunk_doc
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.config import settings


def ingest_document(source_id: str, source_name: str, text: str, metadata: dict | None = None) -> dict:
    chunks = chunk_text(
        text=text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        return {"inserted_chunks": 0}

    embeddings = embed_texts(chunks)

    ops = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        chunk_id = f"{source_id}:{i}"
        doc = make_chunk_doc(
            source_id=source_id,
            source_name=source_name,
            chunk_id=chunk_id,
            text=chunk,
            embedding=embedding,
            metadata=metadata or {},
        )
        ops.append(
            UpdateOne(
                {"chunk_id": chunk_id},
                {"$set": doc},
                upsert=True,
            )
        )

    if ops:
        result = collection.bulk_write(ops, ordered=False)
        return {
            "inserted_chunks": len(chunks),
            "upserted_count": result.upserted_count,
            "modified_count": result.modified_count,
        }

    return {"inserted_chunks": 0}