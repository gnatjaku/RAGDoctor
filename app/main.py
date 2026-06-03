import logging
import logging.config

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response

from app.config import settings
from app.schemas import AskRequest, AskResponse, ExportAnswerPdfRequest, IngestDocument, RetrievedChunk
from app.services.ingest import ingest_document
from app.services.retrieve import retrieve_chunks
from app.services.answer import generate_grounded_answer
from app.services.pdf_export import render_answer_pdf

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        },
    },
    "root": {
        "level": settings.log_level.upper(),
        "handlers": ["console"],
    },
    # Uvicorn loggers – ustaw ten sam poziom i handler
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        # Wyciszenie heartbeatów pymongo – generują zbyt dużo szumu na DEBUG
        "pymongo": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "pymongo.topology": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "app.services": {"handlers": ["console"], "level": settings.log_level.upper(), "propagate": False},
        "app.services.answer": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
        "app.services.embeddings": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
        "app.services.retrieve": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
        "app.services.ingest": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
        "app.services.chunking": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

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
    if (
        doc.chunk_size is not None
        and doc.chunk_overlap is not None
        and doc.chunk_overlap >= doc.chunk_size
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    try:
        result = ingest_document(
            source_id=doc.source_id,
            source_name=doc.source_name,
            text=doc.text,
            metadata=doc.metadata,
            chunk_size=doc.chunk_size,
            chunk_overlap=doc.chunk_overlap,
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


@app.post("/answers/pdf", tags=["ask"])
def export_answer_pdf(req: ExportAnswerPdfRequest) -> Response:
    if not req.answer or not req.answer.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Treść odpowiedzi nie może być pusta.",
        )

    try:
        pdf_bytes, file_name = render_answer_pdf(
            answer=req.answer,
            question=req.question,
            citations=[c.model_dump() for c in req.citations],
            file_name=req.file_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )

