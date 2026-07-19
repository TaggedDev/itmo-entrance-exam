import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ml_service.config import Settings
from ml_service.kb import retrieve_context
from ml_service.schemas import RetrievedContext, TicketClassification, TicketResponse
from ml_service.storage import JsonlStore


DEFAULT_CATEGORIES = ("auth", "feedback", "legal", "payments", "unknown")
SENSITIVE_CATEGORIES = {"legal", "payments"}


class TicketTextPreprocessor:
    email_pattern = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE)
    card_pattern = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    phone_pattern = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
    spaces_pattern = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        return self.spaces_pattern.sub(" ", text.strip().lower())

    def redact(self, text: str) -> str:
        redacted = self.email_pattern.sub("[EMAIL]", text)
        redacted = self.card_pattern.sub("[CARD]", redacted)
        return self.phone_pattern.sub("[PHONE]", redacted)

    def prepare(self, text: str) -> tuple[str, str]:
        normalized = self.normalize(text)
        return normalized, self.redact(normalized)


class DeepSeekChatFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create(self) -> ChatOpenAI | None:
        if not self.settings.deepseek_api_key:
            return None
        return ChatOpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
            temperature=0,
        )


class TicketClassifier:
    def __init__(self, llm: Any, categories: list[str]) -> None:
        self.llm = llm
        self.categories = categories
        self.parser = PydanticOutputParser(pydantic_object=TicketClassification)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You classify support tickets for a small PoC. "
                    "Return only valid JSON. Do not add risk, confidence, or explanation.\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "Allowed categories: {categories}\n"
                    "Use unknown only when none of the categories fit.\n"
                    "Set requires_human_review=true for legal requests, payment/refund disputes, "
                    "account takeover, missing knowledge, or unsafe/uncertain cases.\n\n"
                    "Ticket:\n{ticket_text}",
                ),
            ]
        )

    def classify(self, ticket_text: str) -> TicketClassification:
        if self.llm is None:
            return self._fallback(ticket_text)
        try:
            message = self.prompt.invoke(
                {
                    "categories": ", ".join(self.categories),
                    "ticket_text": ticket_text,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
            raw = self.llm.invoke(message)
            classification = self.parser.parse(str(getattr(raw, "content", raw)))
        except Exception:
            return self._fallback(ticket_text)
        if classification.category not in self.categories:
            return TicketClassification(category="unknown", requires_human_review=True)
        if classification.category in SENSITIVE_CATEGORIES:
            classification.requires_human_review = True
        return classification

    def _fallback(self, ticket_text: str) -> TicketClassification:
        category = "unknown"
        for candidate in self.categories:
            if candidate != "unknown" and candidate in ticket_text:
                category = candidate
                break
        return TicketClassification(
            category=category,
            requires_human_review=category in SENSITIVE_CATEGORIES or category == "unknown",
        )


class KnowledgeRetriever:
    def __init__(self, settings: Settings, top_k: int = 3) -> None:
        self.settings = settings
        self.top_k = top_k

    def retrieve(self, category: str, ticket_text: str) -> list[RetrievedContext]:
        query = f"{category}\n{ticket_text}"
        contexts: list[RetrievedContext] = []
        for context in retrieve_context(self.settings, query, k=self.top_k):
            contexts.append(
                RetrievedContext(
                    domain=str(context.get("domain", "unknown")),
                    source=str(context.get("source", "unknown")),
                    text=str(context.get("text", "")),
                    score=float(str(context.get("score", 0.0))),
                )
            )
        return contexts


class AnswerGenerator:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You draft concise Russian support replies for a PoC. "
                    "Use only the provided knowledge context. "
                    "If context is insufficient, say that the ticket needs operator review. "
                    "Do not ask for full card numbers, passwords, CVC/CVV, tokens, or 2FA codes.",
                ),
                (
                    "human",
                    "Category: {category}\n"
                    "Ticket: {ticket_text}\n\n"
                    "Knowledge context:\n{context}\n\n"
                    "Draft the answer:",
                ),
            ]
        )

    def generate(self, category: str, ticket_text: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "Недостаточно данных в базе знаний. Передайте тикет оператору для ручной проверки."
        context_text = self._format_context(contexts)
        if self.llm is None:
            return self._fallback_answer(category, contexts)
        try:
            message = self.prompt.invoke(
                {
                    "category": category,
                    "ticket_text": ticket_text,
                    "context": context_text,
                }
            )
            raw = self.llm.invoke(message)
            answer = str(getattr(raw, "content", raw)).strip()
        except Exception:
            return self._fallback_answer(category, contexts)
        return answer or self._fallback_answer(category, contexts)

    def _format_context(self, contexts: list[RetrievedContext]) -> str:
        return "\n\n".join(
            f"Source: {context.source}\nDomain: {context.domain}\nText: {context.text}"
            for context in contexts
        )

    def _fallback_answer(self, category: str, contexts: list[RetrievedContext]) -> str:
        first = contexts[0]
        return (
            f"Черновик ответа по категории '{category}'. "
            f"Используйте источник {first.source}: {first.text[:500]}"
        )


class TicketPipeline:
    def __init__(
        self,
        settings: Settings,
        store: JsonlStore,
        preprocessor: TicketTextPreprocessor | None = None,
        classifier: TicketClassifier | None = None,
        retriever: KnowledgeRetriever | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        llm = DeepSeekChatFactory(settings).create()
        categories = self._load_categories(settings.knowledge_dir)
        self.preprocessor = preprocessor or TicketTextPreprocessor()
        self.classifier = classifier or TicketClassifier(llm=llm, categories=categories)
        self.retriever = retriever or KnowledgeRetriever(settings=settings)
        self.answer_generator = answer_generator or AnswerGenerator(llm=llm)

    def process(self, text: str, channel: str, user_id: str | None) -> dict[str, object]:
        normalized_text, redacted_text = self.preprocessor.prepare(text)
        classification = self.classifier.classify(redacted_text)
        contexts = self.retriever.retrieve(classification.category, redacted_text)
        requires_human_review = classification.requires_human_review or not contexts
        answer = self.answer_generator.generate(classification.category, redacted_text, contexts)
        sources = sorted({context.source for context in contexts})
        decision = "needs_human_review" if requires_human_review else "auto_draft_ready"
        result = TicketResponse(
            ticket_id=str(uuid.uuid4()),
            original_text=text,
            normalized_text=normalized_text,
            redacted_text=redacted_text,
            category=classification.category,
            requires_human_review=requires_human_review,
            retrieved_context=contexts,
            answer=answer,
            sources=sources,
            decision=decision,
            llm_provider=self.settings.deepseek_model,
            langfuse_enabled=bool(self.settings.langfuse_public_key and self.settings.langfuse_secret_key),
        )
        payload = result.model_dump()
        self.store.append_audit({"event": "processed", "channel": channel, "user_id": user_id, **payload})
        if requires_human_review:
            self.store.append_pending(payload)
        return payload

    def _load_categories(self, knowledge_dir: Path) -> list[str]:
        categories = sorted(path.stem for path in knowledge_dir.glob("*.txt"))
        if "unknown" not in categories:
            categories.append("unknown")
        return categories or list(DEFAULT_CATEGORIES)


def process_ticket(settings: Settings, store: JsonlStore, text: str, channel: str, user_id: str | None) -> dict[str, object]:
    return TicketPipeline(settings=settings, store=store).process(text=text, channel=channel, user_id=user_id)
