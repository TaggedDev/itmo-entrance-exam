from pathlib import Path

from fastapi.testclient import TestClient

from ml_service.config import get_settings
from ml_service.main import app


client = TestClient(app)


def test_health_reports_deepseek_key_status() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "deepseek_api_key_loaded" in response.json()
    assert response.json()["embedding_provider"] == "hash"


def test_process_safe_ticket_returns_draft() -> None:
    response = client.post(
        "/tickets/process",
        json={"text": "Не приходит код для входа в аккаунт", "channel": "web"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "auth"
    assert payload["decision"] == "auto_draft_ready"
    assert payload["draft_response"]


def test_process_risky_ticket_goes_to_moderation() -> None:
    response = client.post(
        "/tickets/process",
        json={"text": "У меня юридическая претензия и я пойду в суд", "channel": "email"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_level"] == "high"
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
