from fastapi import FastAPI, HTTPException

from ml_service.config import get_settings
from ml_service.kb import inspect_knowledge, reindex_knowledge_async
from ml_service.pipeline import process_ticket_async
from ml_service.schemas import (
    KnowledgeInspectResponse,
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
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
    }


@app.post("/knowledge/reindex", response_model=ReindexResponse)
async def reindex() -> ReindexResponse:
    with tracer.observation("knowledge_reindex", as_type="chain", metadata={"collection": settings.chroma_collection}):
        indexed_files, indexed_chunks = await reindex_knowledge_async(settings, tracer=tracer)
    tracer.flush()
    return ReindexResponse(
        indexed_files=indexed_files,
        indexed_chunks=indexed_chunks,
        collection=settings.chroma_collection,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
    )


@app.get("/knowledge/inspect", response_model=KnowledgeInspectResponse)
def inspect(limit: int = 10) -> dict[str, object]:
    return inspect_knowledge(settings, limit=max(1, min(limit, 50)))


@app.post("/tickets/process", response_model=TicketResponse)
async def process(request: TicketRequest) -> dict[str, object]:
    with tracer.observation(
        "ticket_process",
        as_type="chain",
        input={"channel": request.channel, "characters": len(request.text), "user_id": request.user_id},
    ) as observation:
        result = await process_ticket_async(settings, store, request.text, request.channel, request.user_id, tracer=tracer)
        observation.update(
            output={
                "ticket_id": result["ticket_id"],
                "category": result["category"],
                "decision": result["decision"],
                "sources": result["sources"],
            }
        )
    tracer.flush()
    return result


@app.get("/tickets/pending", response_model=list[PendingTicket])
async def pending() -> list[dict[str, object]]:
    return await store.list_pending_async()


@app.post("/tickets/{ticket_id}/moderate", response_model=ModerationResponse)
async def moderate(ticket_id: str, request: ModerationRequest) -> ModerationResponse:
    if request.action not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="action must be approve or reject")
    status = await store.moderate_async(ticket_id, request.action, request.operator_note)
    if status == "not_found":
        raise HTTPException(status_code=404, detail="ticket not found")
    return ModerationResponse(ticket_id=ticket_id, status=status)
