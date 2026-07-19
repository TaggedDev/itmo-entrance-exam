import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml_service.config import get_settings
from ml_service.main import app
from ml_service.pipeline import AnswerGenerator, TicketClassifier, TicketPipeline
from ml_service.schemas import RetrievedContext, TicketClassification
from ml_service.storage import JsonlStore


client = TestClient(app)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class ContractLLM:
    def __init__(self, category: str, requires_human_review: bool) -> None:
        self.category = category
        self.requires_human_review = requires_human_review
        self.calls = 0

    def invoke(self, _message: object) -> FakeMessage:
        return self._next_message()

    async def ainvoke(self, _message: object) -> FakeMessage:
        return self._next_message()

    def _next_message(self) -> FakeMessage:
        self.calls += 1
        if self.calls == 1:
            review = "true" if self.requires_human_review else "false"
            return FakeMessage(f'{{"category": "{self.category}", "requires_human_review": {review}}}')
        return FakeMessage("Контрактный ответ на основе базы знаний.")


class BrokenLLM:
    def invoke(self, _message: object) -> FakeMessage:
        raise RuntimeError("LLM unavailable")

    async def ainvoke(self, _message: object) -> FakeMessage:
        raise RuntimeError("LLM unavailable")


class FixedClassifier:
    def __init__(self, category: str, requires_human_review: bool) -> None:
        self.category = category
        self.requires_human_review = requires_human_review

    def classify(self, _ticket_text: str) -> TicketClassification:
        return TicketClassification(category=self.category, requires_human_review=self.requires_human_review)


class FixedRetriever:
    def __init__(self, contexts: list[RetrievedContext]) -> None:
        self.contexts = contexts

    def retrieve(self, _category: str, _ticket_text: str) -> list[RetrievedContext]:
        return self.contexts


def auth_context() -> RetrievedContext:
    return RetrievedContext(
        domain="auth",
        source="auth.txt",
        text="Если код входа не приходит, проверьте папку спам и запросите новый код через 60 секунд.",
        score=0.91,
    )


async def fake_contexts(_settings: object, _query: str, k: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "domain": "auth",
            "source": "auth.txt",
            "text": "Если код входа не приходит, можно запросить новый код.",
            "score": 0.91,
        }
    ][:k]


def test_safe_ticket_keeps_public_response_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ml_service.main.store", JsonlStore(tmp_path))
    monkeypatch.setattr("ml_service.pipeline.DeepSeekChatFactory.create", lambda _factory: ContractLLM("auth", False))
    monkeypatch.setattr("ml_service.pipeline.retrieve_context_async", fake_contexts)

    response = client.post("/tickets/process", json={"text": "Не приходит код для входа", "channel": "web"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "auth"
    assert payload["requires_human_review"] is False
    assert payload["decision"] == "auto_draft_ready"
    assert payload["answer"]
    assert payload["sources"] == ["auth.txt"]
    assert payload["retrieved_context"][0]["source"] == "auth.txt"


def test_risky_ticket_goes_to_pending_moderation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr("ml_service.main.store", store)
    monkeypatch.setattr("ml_service.pipeline.DeepSeekChatFactory.create", lambda _factory: ContractLLM("legal", True))
    monkeypatch.setattr("ml_service.pipeline.retrieve_context_async", fake_contexts)

    response = client.post("/tickets/process", json={"text": "Я подам иск в суд", "channel": "email"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["requires_human_review"] is True
    assert payload["decision"] == "needs_human_review"
    assert any(ticket["ticket_id"] == payload["ticket_id"] for ticket in store.list_pending())


@pytest.mark.asyncio
async def test_no_context_non_general_ticket_goes_to_human_review(tmp_path: Path) -> None:
    pipeline = TicketPipeline(
        settings=get_settings(),
        store=JsonlStore(tmp_path),
        classifier=FixedClassifier("auth", False),
        retriever=FixedRetriever([]),
        answer_generator=AnswerGenerator(llm=None),
    )

    payload = await pipeline.aprocess("Не приходит код для входа", "web", "1234")

    assert payload["requires_human_review"] is True
    assert payload["decision"] == "needs_human_review"
    assert payload["answer"]


@pytest.mark.asyncio
async def test_llm_unavailable_uses_fallback_without_crashing(tmp_path: Path) -> None:
    pipeline = TicketPipeline(
        settings=get_settings(),
        store=JsonlStore(tmp_path),
        classifier=TicketClassifier(
            llm=BrokenLLM(),
            categories=["auth", "feedback", "general", "legal", "payments", "unknown"],
        ),
        retriever=FixedRetriever([auth_context()]),
        answer_generator=AnswerGenerator(llm=BrokenLLM()),
    )

    payload = await pipeline.aprocess("auth: Не приходит код для входа", "web", "1234")

    assert payload["category"] == "auth"
    assert payload["decision"] == "auto_draft_ready"
    assert payload["answer"]


@pytest.mark.asyncio
async def test_audit_event_records_ticket_category_and_decision(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    pipeline = TicketPipeline(
        settings=get_settings(),
        store=store,
        classifier=FixedClassifier("auth", False),
        retriever=FixedRetriever([auth_context()]),
        answer_generator=AnswerGenerator(llm=None),
    )

    payload = await pipeline.aprocess("Не приходит код для входа", "web", "1234")

    events = [json.loads(line) for line in store.audit_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["ticket_id"] == payload["ticket_id"]
    assert events[-1]["category"] == "auth"
    assert events[-1]["decision"] == "auto_draft_ready"


def test_reindex_response_contract_and_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_reindex(_settings: object, tracer: object | None = None) -> tuple[int, int]:
        return 2, 3

    def fake_inspect(_settings: object, limit: int = 10) -> dict[str, object]:
        return {
            "collection": "support_kb",
            "count": 3,
            "embedding_provider": "hash",
            "embedding_model": "intfloat/multilingual-e5-small",
            "items": [
                {
                    "id": "chunk-1",
                    "metadata": {"domain": "auth", "source": "auth.txt", "chunk": 0},
                    "text": "Auth knowledge",
                    "characters": 14,
                    "embedding_dimensions": 16,
                    "embedding_preview": [0.1, 0.2],
                }
            ][:limit],
        }

    monkeypatch.setattr("ml_service.main.reindex_knowledge_async", fake_reindex)
    monkeypatch.setattr("ml_service.main.inspect_knowledge", fake_inspect)

    response = client.post("/knowledge/reindex")
    assert response.status_code == 200
    assert response.json()["indexed_files"] == 2
    assert response.json()["indexed_chunks"] == 3

    inspect_response = client.get("/knowledge/inspect?limit=1")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["count"] == 3
    assert inspect_payload["items"][0]["embedding_dimensions"] > 0
