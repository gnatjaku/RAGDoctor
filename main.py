from fastapi import FastAPI, HTTPException
from app.schemas import IngestDocument, AskRequest, AskResponse, RetrievedChunk
from app.services.ingest import ingest_document
from app.services.retrieve import retrieve_chunks
from app.services.answer import generate_grounded_answer

app = FastAPI(title="RAG API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(doc: IngestDocument) -> dict:
    if not doc.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    result = ingest_document(
        source_id=doc.source_id,
        source_name=doc.source_name,
        text=doc.text,
        metadata=doc.metadata,
    )
    return {"status": "ok", **result}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    chunks = retrieve_chunks(req.question, req.top_k)
    if not chunks:
        return AskResponse(answer="I don't know.", citations=[])

    answer = generate_grounded_answer(req.question, chunks)

    citations = [
        RetrievedChunk(
            source_name=c["source_name"],
            chunk_id=c["chunk_id"],
            score=float(c["score"]),
            text=c["text"],
            metadata=c.get("metadata", {}),
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, citations=citations)