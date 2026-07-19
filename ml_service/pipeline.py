import uuid

from ml_service.config import Settings
from ml_service.kb import retrieve_context
from ml_service.storage import JsonlStore


RISKY_KEYWORDS = {
    "юрид": "legal",
    "суд": "legal",
    "претензи": "legal",
    "персональн": "legal",
    "удалить данные": "legal",
    "возврат": "payments",
    "верните деньги": "payments",
    "списали дважды": "payments",
    "взлом": "auth",
    "украли": "auth",
}

CATEGORY_KEYWORDS = {
    "payments": ["оплат", "платеж", "списан", "подписк", "возврат", "деньги"],
    "auth": ["вход", "логин", "парол", "код", "email", "аккаунт"],
    "feedback": ["предлож", "улучш", "фича", "идея", "хочу"],
    "legal": ["юрид", "суд", "претензи", "персональн", "данные"],
}


def process_ticket(settings: Settings, store: JsonlStore, text: str, channel: str, user_id: str | None) -> dict[str, object]:
    ticket_id = str(uuid.uuid4())
    category, confidence = classify(text)
    risk_level = assess_risk(text, confidence)
    contexts = retrieve_context(settings, text)
    decision = "needs_human_review" if risk_level == "high" or confidence < 0.65 else "auto_draft_ready"
    draft_response = generate_mock_response(category, risk_level, contexts)
    result = {
        "ticket_id": ticket_id,
        "text": text,
        "channel": channel,
        "user_id": user_id,
        "category": category,
        "risk_level": risk_level,
        "confidence": confidence,
        "retrieved_context": contexts,
        "draft_response": draft_response,
        "decision": decision,
        "llm_provider": "mock-deepseek",
        "langfuse_enabled": bool(settings.langfuse_public_key and settings.langfuse_secret_key),
    }
    store.append_audit({"event": "processed", **result})
    if decision == "needs_human_review":
        store.append_pending(result)
    return result


def classify(text: str) -> tuple[str, float]:
    normalized = text.lower()
    scores = {
        category: sum(1 for keyword in keywords if keyword in normalized)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    category = max(scores, key=scores.get)
    best_score = scores[category]
    if best_score == 0:
        return "unknown", 0.4
    return category, min(0.95, 0.55 + best_score * 0.15)


def assess_risk(text: str, confidence: float) -> str:
    normalized = text.lower()
    if any(keyword in normalized for keyword in RISKY_KEYWORDS):
        return "high"
    if confidence < 0.65:
        return "medium"
    return "low"


def generate_mock_response(category: str, risk_level: str, contexts: list[dict[str, object]]) -> str:
    if risk_level == "high":
        return (
            "Черновик для оператора: обращение относится к рискованной категории. "
            "Не отправляйте ответ пользователю без проверки специалистом."
        )
    context_hint = contexts[0]["text"] if contexts else "релевантный фрагмент базы знаний не найден"
    return (
        f"Здравствуйте! Мы проверили обращение в категории '{category}'. "
        f"Рекомендация из базы знаний: {str(context_hint)[:300]}"
    )


def build_deepseek_payload(prompt: str, settings: Settings) -> dict[str, object]:
    return {
        "model": settings.deepseek_model,
        "messages": [{"role": "user", "content": prompt}],
        "api_key_present": bool(settings.deepseek_api_key),
    }
