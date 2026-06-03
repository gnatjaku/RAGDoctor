from pydantic import BaseModel, Field
from typing import Any


class IngestDocument(BaseModel):
    source_id: str = Field(..., description="Stable document identifier")
    source_name: str = Field(..., description="Human-readable file or doc name")
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int | None = Field(default=None, ge=1, description="Optional chunk size override for this document")
    chunk_overlap: int | None = Field(default=None, ge=0, description="Optional chunk overlap override for this document")


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


class ExportAnswerPdfRequest(BaseModel):
    answer: str
    question: str | None = None
    citations: list[RetrievedChunk] = Field(default_factory=list)
    file_name: str | None = None

