from fastapi import FastAPI, HTTPException, status

from app.schemas import AskRequest, AskResponse, IngestDocument, RetrievedChunk
from app.services.ingest import ingest_document
from app.services.retrieve import retrieve_chunks
from app.services.answer import generate_grounded_answer

app = FastAPI(
    title="RAGDoctor",
    description="Retrieval-Augmented Generation API for medical documents",
    version="0.1.0",
)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", status_code=status.HTTP_201_CREATED, tags=["ingest"])
def ingest(doc: IngestDocument) -> dict:
    """
    Chunk, embed and upsert a document into MongoDB.
    """
    try:
        result = ingest_document(
            source_id=doc.source_id,
            source_name=doc.source_name,
            text=doc.text,
            metadata=doc.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/ask", response_model=AskResponse, tags=["ask"])
def ask(req: AskRequest) -> AskResponse:
    """
    Retrieve relevant chunks and generate a grounded answer.
    """
    try:
        chunks = retrieve_chunks(question=req.question, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks found. Try ingesting documents first.",
        )

    try:
        answer = generate_grounded_answer(question=req.question, chunks=chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    citations = [RetrievedChunk(**c) for c in chunks]
    return AskResponse(answer=answer, citations=citations)
