from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ml_service.config import get_settings
from ml_service.main import app
from ml_service.pipeline import TicketTextPreprocessor


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


def fake_contexts(_settings: object, _query: str, k: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "domain": "auth",
            "source": "auth.txt",
            "text": "Инструкция из базы знаний.",
            "score": 0.9,
        }
    ][:k]


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


def test_process_safe_ticket_returns_answer(monkeypatch) -> None:
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


def test_process_review_ticket_goes_to_moderation(monkeypatch) -> None:
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
