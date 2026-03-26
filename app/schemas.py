from pydantic import BaseModel, Field
from typing import Any


class IngestDocument(BaseModel):
    source_id: str = Field(..., description="Stable document identifier")
    source_name: str = Field(..., description="Human-readable file or doc name")
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class RetrievedChunk(BaseModel):
    source_name: str
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    citations: list[RetrievedChunk]