from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    text: str = Field(min_length=1)
    channel: str = "web"
    user_id: str | None = None


class TicketClassification(BaseModel):
    category: str
    requires_human_review: bool


class RetrievedContext(BaseModel):
    domain: str
    source: str
    text: str
    score: float


class TicketResponse(BaseModel):
    ticket_id: str
    original_text: str
    normalized_text: str
    redacted_text: str
    category: str
    requires_human_review: bool
    retrieved_context: list[RetrievedContext]
    answer: str
    sources: list[str]
    decision: str
    llm_provider: str
    langfuse_enabled: bool


class ReindexResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    collection: str
    embedding_provider: str
    embedding_model: str


class KnowledgeChunkPreview(BaseModel):
    id: str
    metadata: dict[str, object]
    text: str
    characters: int
    embedding_dimensions: int
    embedding_preview: list[float]


class KnowledgeInspectResponse(BaseModel):
    collection: str
    count: int
    embedding_provider: str
    embedding_model: str
    items: list[KnowledgeChunkPreview]


class PendingTicket(BaseModel):
    ticket_id: str
    original_text: str
    redacted_text: str
    category: str
    requires_human_review: bool
    decision: str
    answer: str


class ModerationRequest(BaseModel):
    action: str
    operator_note: str = ""


class ModerationResponse(BaseModel):
    ticket_id: str
    status: str
