from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    text: str = Field(min_length=1)
    channel: str = "web"
    user_id: str | None = None


class RetrievedContext(BaseModel):
    domain: str
    text: str
    score: float


class TicketResponse(BaseModel):
    ticket_id: str
    category: str
    risk_level: str
    confidence: float
    retrieved_context: list[RetrievedContext]
    draft_response: str
    decision: str
    llm_provider: str
    langfuse_enabled: bool


class ReindexResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    collection: str


class PendingTicket(BaseModel):
    ticket_id: str
    text: str
    category: str
    risk_level: str
    decision: str
    draft_response: str


class ModerationRequest(BaseModel):
    action: str
    operator_note: str = ""


class ModerationResponse(BaseModel):
    ticket_id: str
    status: str
