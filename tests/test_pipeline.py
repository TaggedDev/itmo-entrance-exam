from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ml_service.config import get_settings
from ml_service.main import app
from ml_service.pipeline import AnswerGenerator, TicketClassifier, TicketPipeline, TicketTextPreprocessor
from ml_service.schemas import RetrievedContext, TicketClassification
from ml_service.storage import JsonlStore


client = TestClient(app)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, category: str, requires_human_review: bool) -> None:
        self.category = category
        self.requires_human_review = requires_human_review
        self.calls = 0

    def invoke(self, _message: object) -> FakeMessage:
        self.calls += 1
        if self.calls == 1:
            review = "true" if self.requires_human_review else "false"
            return FakeMessage(f'{{"category": "{self.category}", "requires_human_review": {review}}}')
        return FakeMessage("Тестовый ответ на основе базы знаний.")


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


def fake_contexts(_settings: object, _query: str, k: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "domain": "auth",
            "source": "auth.txt",
            "text": "Инструкция из базы знаний.",
            "score": 0.9,
        }
    ][:k]


def payment_context() -> RetrievedContext:
    return RetrievedContext(
        domain="payments",
        source="payments.txt",
        text="Подписка стоит 10 долларов в месяц.",
        score=0.95,
    )


def make_pipeline(
    tmp_path: Path,
    classifier: object,
    contexts: list[RetrievedContext],
) -> TicketPipeline:
    return TicketPipeline(
        settings=get_settings(),
        store=JsonlStore(tmp_path),
        classifier=classifier,
        retriever=FixedRetriever(contexts),
        answer_generator=AnswerGenerator(llm=None),
    )


def test_preprocessor_normalizes_and_redacts_pii() -> None:
    preprocessor = TicketTextPreprocessor()

    normalized, redacted = preprocessor.prepare(
        "  User TEST@Example.COM paid with 4111 1111 1111 1111 and phone +7 999 123-45-67  "
    )

    assert normalized == "user test@example.com paid with 4111 1111 1111 1111 and phone +7 999 123-45-67"
    assert "[EMAIL]" in redacted
    assert "[CARD]" in redacted
    assert "[PHONE]" in redacted


def test_health_reports_deepseek_key_status() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "deepseek_api_key_loaded" in response.json()
    assert response.json()["embedding_provider"] == "hash"


def test_process_safe_ticket_returns_answer(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ml_service.main.store", JsonlStore(tmp_path))
    monkeypatch.setattr("ml_service.pipeline.DeepSeekChatFactory.create", lambda _factory: FakeLLM("auth", False))
    monkeypatch.setattr("ml_service.pipeline.retrieve_context", fake_contexts)

    response = client.post(
        "/tickets/process",
        json={"text": "Не приходит код для входа в аккаунт", "channel": "web"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "auth"
    assert payload["requires_human_review"] is False
    assert payload["decision"] == "auto_draft_ready"
    assert payload["answer"]
    assert payload["sources"] == ["auth.txt"]
    assert payload["retrieved_context"]


def test_process_review_ticket_goes_to_moderation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ml_service.main.store", JsonlStore(tmp_path))
    monkeypatch.setattr("ml_service.pipeline.DeepSeekChatFactory.create", lambda _factory: FakeLLM("legal", True))
    monkeypatch.setattr("ml_service.pipeline.retrieve_context", fake_contexts)

    response = client.post(
        "/tickets/process",
        json={"text": "У меня юридическая претензия", "channel": "email"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "legal"
    assert payload["requires_human_review"] is True
    assert payload["decision"] == "needs_human_review"

    pending = client.get("/tickets/pending")
    assert pending.status_code == 200
    assert any(ticket["ticket_id"] == payload["ticket_id"] for ticket in pending.json())


def test_greeting_does_not_go_to_moderation(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    pipeline = TicketPipeline(
        settings=get_settings(),
        store=store,
        classifier=TicketClassifier(
            llm=None,
            categories=["auth", "feedback", "general", "legal", "payments", "unknown"],
        ),
        retriever=FixedRetriever([]),
        answer_generator=AnswerGenerator(llm=None),
    )

    payload = pipeline.process("Привет", "web", "1234")

    assert payload["category"] == "general"
    assert payload["requires_human_review"] is False
    assert payload["decision"] == "auto_draft_ready"
    assert store.list_pending() == []


def test_known_payment_fact_does_not_require_moderation(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, FixedClassifier("payments", False), [payment_context()])

    payload = pipeline.process("Сколько стоит подписка?", "web", "1234")

    assert payload["category"] == "payments"
    assert payload["requires_human_review"] is False
    assert payload["decision"] == "auto_draft_ready"


def test_payment_dispute_requires_moderation(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    pipeline = TicketPipeline(
        settings=get_settings(),
        store=store,
        classifier=FixedClassifier("payments", True),
        retriever=FixedRetriever([payment_context()]),
        answer_generator=AnswerGenerator(llm=None),
    )

    payload = pipeline.process("Списали деньги дважды, верните деньги", "web", "1234")

    assert payload["requires_human_review"] is True
    assert payload["decision"] == "needs_human_review"
    assert any(ticket["ticket_id"] == payload["ticket_id"] for ticket in store.list_pending())


def test_legal_claim_requires_moderation(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, FixedClassifier("legal", True), [payment_context()])

    payload = pipeline.process("Я подам иск в суд", "email", "1234")

    assert payload["category"] == "legal"
    assert payload["requires_human_review"] is True
    assert payload["decision"] == "needs_human_review"


def test_non_general_without_context_requires_moderation(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, FixedClassifier("auth", False), [])

    payload = pipeline.process("Не приходит код для входа", "web", "1234")

    assert payload["requires_human_review"] is True
    assert payload["decision"] == "needs_human_review"


def test_reindex_uses_knowledge_directory() -> None:
    settings = get_settings()
    assert Path(settings.knowledge_dir).exists()
    response = client.post("/knowledge/reindex")
    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed_files"] >= 4
    assert payload["indexed_chunks"] >= 4
    assert payload["embedding_provider"] == "hash"

    inspect_response = client.get("/knowledge/inspect?limit=2")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["count"] >= payload["indexed_chunks"]
    assert len(inspect_payload["items"]) <= 2
    assert inspect_payload["items"][0]["embedding_dimensions"] > 0
    assert inspect_payload["items"][0]["embedding_preview"]
