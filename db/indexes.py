"""
MongoDB index setup for production.

Indexes created:
  1. Unique index on `chunk_id`  – prevents duplicate chunks.
  2. Atlas Vector Search index on `embedding` – enables $vectorSearch queries.

Usage:
    python -m db.indexes
    # or import and call from app startup:
    from db.indexes import create_indexes
    create_indexes()
"""

from __future__ import annotations

import logging
import os

from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings (read from env / .env)
# ---------------------------------------------------------------------------
MONGO_URI: str = os.getenv(
    "MONGO_URI",
    "mongodb://{user}:{password}@localhost:27017".format(
        user=os.getenv("MONGO_INITDB_ROOT_USERNAME", "admin"),
        password=os.getenv("MONGO_INITDB_ROOT_PASSWORD", ""),
    ),
)
DB_NAME: str = os.getenv("MONGO_DB_NAME", "ragdoctor")
COLLECTION_NAME: str = os.getenv("MONGO_COLLECTION_NAME", "chunks")

# Vector Search settings
EMBEDDING_FIELD: str = "embedding"
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))  # OpenAI ada-002 default
VECTOR_SEARCH_INDEX_NAME: str = "vector_index"
SIMILARITY: str = "cosine"  # "cosine" | "euclidean" | "dotProduct"


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------

def _create_unique_chunk_id_index(collection) -> None:
    """Unique index on chunk_id – prevents duplicate document chunks."""
    index_name = "chunk_id_unique"
    existing = [idx["name"] for idx in collection.list_indexes()]

    if index_name in existing:
        logger.info("Index '%s' already exists – skipping.", index_name)
        return

    collection.create_index(
        [("chunk_id", ASCENDING)],
        unique=True,
        name=index_name,
        background=True,  # non-blocking build on existing data
    )
    logger.info("Created unique index '%s' on field 'chunk_id'.", index_name)


def _create_vector_search_index(collection) -> None:
    """
    Atlas Vector Search index on the `embedding` field.

    On MongoDB Atlas:
      - Creates a proper $vectorSearch index (type: vectorSearch).
      - Enables ANN similarity search via the $vectorSearch aggregation stage.

    On self-hosted / local MongoDB (e.g. Docker):
      - list_search_indexes() / create_search_index() raise OperationFailure
        because $listSearchIndexes / $createSearchIndex are Atlas-only.
      - Falls back gracefully to a plain ascending index on the embedding field,
        which is useful for local development but does NOT support $vectorSearch.
      - To get real vector search locally, use MongoDB Atlas or
        MongoDB 7.0+ with the Atlas Search Local feature.
    """
    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": EMBEDDING_FIELD,
                "numDimensions": EMBEDDING_DIMENSIONS,
                "similarity": SIMILARITY,
            }
        ]
    }

    try:
        existing_search_indexes = [
            idx["name"]
            for idx in collection.list_search_indexes()
        ]

        if VECTOR_SEARCH_INDEX_NAME in existing_search_indexes:
            logger.info(
                "Vector Search index '%s' already exists – skipping.",
                VECTOR_SEARCH_INDEX_NAME,
            )
            return

        collection.create_search_index(
            {
                "name": VECTOR_SEARCH_INDEX_NAME,
                "type": "vectorSearch",
                "definition": index_definition,
            }
        )
        logger.info(
            "Created Atlas Vector Search index '%s' on field '%s' "
            "(dims=%d, similarity=%s).",
            VECTOR_SEARCH_INDEX_NAME,
            EMBEDDING_FIELD,
            EMBEDDING_DIMENSIONS,
            SIMILARITY,
        )
    except OperationFailure as exc:
        # Self-hosted MongoDB without Atlas Search support
        logger.warning(
            "Atlas Vector Search not available on this deployment "
            "(requires MongoDB Atlas). Error: %s\n"
            "Falling back to a plain ascending index on '%s' for local dev.\n"
            "Deploy to Atlas or use mongot sidecar for full $vectorSearch support.",
            exc,
            EMBEDDING_FIELD,
        )
        # Fallback: plain index for local dev – does NOT support $vectorSearch
        fallback_name = f"{EMBEDDING_FIELD}_plain"
        existing = [idx["name"] for idx in collection.list_indexes()]
        if fallback_name not in existing:
            collection.create_index(
                [(EMBEDDING_FIELD, ASCENDING)],
                name=fallback_name,
                sparse=True,
            )
            logger.info("Created fallback plain index '%s'.", fallback_name)


def create_indexes(
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Entry point – create all production indexes.

    Parameters
    ----------
    mongo_uri:        MongoDB connection string.
    db_name:          Target database name.
    collection_name:  Target collection name.
    """
    client: MongoClient = MongoClient(mongo_uri)
    try:
        db = client[db_name]
        collection = db[collection_name]

        logger.info(
            "Setting up indexes on %s.%s …", db_name, collection_name
        )
        _create_unique_chunk_id_index(collection)
        _create_vector_search_index(collection)
        logger.info("Index setup complete.")
    finally:
        client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        stream=sys.stdout,
    )

    create_indexes()
