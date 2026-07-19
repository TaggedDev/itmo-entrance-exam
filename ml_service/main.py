from fastapi import FastAPI, HTTPException

from ml_service.config import get_settings
from ml_service.kb import reindex_knowledge
from ml_service.pipeline import process_ticket
from ml_service.schemas import (
    ModerationRequest,
    ModerationResponse,
    PendingTicket,
    ReindexResponse,
    TicketRequest,
    TicketResponse,
)
from ml_service.storage import JsonlStore
from ml_service.tracing import LangfuseTracer

app = FastAPI(title="Support Ticket ML Service", version="0.1.0")
settings = get_settings()
store = JsonlStore(settings.data_dir)
tracer = LangfuseTracer(settings)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "deepseek_api_key_loaded": bool(settings.deepseek_api_key),
        "langfuse_enabled": tracer.enabled,
        "chroma_host": settings.chroma_host,
        "chroma_port": settings.chroma_port,
    }


@app.post("/knowledge/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    with tracer.span("knowledge_reindex"):
        indexed_files, indexed_chunks = reindex_knowledge(settings)
    tracer.flush()
    return ReindexResponse(
        indexed_files=indexed_files,
        indexed_chunks=indexed_chunks,
        collection=settings.chroma_collection,
    )


@app.post("/tickets/process", response_model=TicketResponse)
def process(request: TicketRequest) -> dict[str, object]:
    with tracer.span("ticket_process", channel=request.channel):
        result = process_ticket(settings, store, request.text, request.channel, request.user_id)
    tracer.flush()
    return result


@app.get("/tickets/pending", response_model=list[PendingTicket])
def pending() -> list[dict[str, object]]:
    return store.list_pending()


@app.post("/tickets/{ticket_id}/moderate", response_model=ModerationResponse)
def moderate(ticket_id: str, request: ModerationRequest) -> ModerationResponse:
    if request.action not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="action must be approve or reject")
    status = store.moderate(ticket_id, request.action, request.operator_note)
    if status == "not_found":
        raise HTTPException(status_code=404, detail="ticket not found")
    return ModerationResponse(ticket_id=ticket_id, status=status)
