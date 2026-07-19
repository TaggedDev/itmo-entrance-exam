import asyncio
import inspect
import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ml_service.config import Settings
from ml_service.kb import retrieve_context, retrieve_context_async
from ml_service.schemas import RetrievedContext, TicketClassification, TicketResponse
from ml_service.storage import JsonlStore
from ml_service.tracing import LangfuseTracer


_DEFAULT_RETRIEVE_CONTEXT = retrieve_context
DEFAULT_CATEGORIES = ("auth", "feedback", "general", "legal", "payments", "unknown")
GENERAL_CATEGORY = "general"
UNKNOWN_CATEGORY = "unknown"
SMALL_TALK_PATTERNS = (
    "привет",
    "здравствуйте",
    "добрый день",
    "доброе утро",
    "добрый вечер",
    "как дела",
    "hello",
    "hi",
)
REVIEW_KEYWORDS = (
    "суд",
    "иск",
    "претензи",
    "юрист",
    "адвокат",
    "регулятор",
    "роскомнадзор",
    "персональн",
    "удалите данные",
    "верните деньги",
    "возврат",
    "списали",
    "списание",
    "дважды",
    "chargeback",
    "чарджбэк",
    "мошен",
    "украли",
    "захват",
    "взлом",
)


async def _call_llm_async(llm: Any, message: object) -> object:
    if hasattr(llm, "ainvoke"):
        result = llm.ainvoke(message)
        if inspect.isawaitable(result):
            return await result
        return result
    return await asyncio.to_thread(llm.invoke, message)


async def _retrieve_context_compatible(settings: Settings, query: str, k: int) -> list[dict[str, object]]:
    if retrieve_context is not _DEFAULT_RETRIEVE_CONTEXT:
        return await asyncio.to_thread(retrieve_context, settings, query, k)
    return await retrieve_context_async(settings, query, k=k)


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
    def __init__(self, llm: Any, categories: list[str], tracer: LangfuseTracer | None = None) -> None:
        self.llm = llm
        self.categories = categories
        self.tracer = tracer
        self.parser = PydanticOutputParser(pydantic_object=TicketClassification)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты классифицируешь обращения пользователей для небольшого PoC поддержки. "
                    "Верни только валидный JSON по схеме. Не добавляй risk, confidence или объяснения.\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "Доступные категории: {categories}\n"
                    "Используй unknown только если ни одна категория не подходит.\n"
                    "Используй general для приветствий, small talk и сообщений без конкретной проблемы.\n\n"
                    "Правила human review:\n"
                    "- true: юридическая претензия, суд, регулятор, privacy-запрос, официальный запрос, "
                    "спорное или некорректное списание, возврат денег, chargeback/fraud, захват аккаунта, "
                    "небезопасный запрос или явная неопределенность.\n"
                    "- false: приветствие, small talk, обычный отзыв, типовой вопрос или простой факт из базы знаний.\n"
                    "- Тема денег сама по себе не требует проверки человеком, если пользователь спрашивает известный факт "
                    "без спора, требования или жалобы, например стоимость подписки.\n\n"
                    "Тикет:\n{ticket_text}",
                ),
            ]
        )

    def classify(self, ticket_text: str) -> TicketClassification:
        if self._is_small_talk(ticket_text):
            return TicketClassification(category=GENERAL_CATEGORY, requires_human_review=False)
        if self.llm is None:
            return self._fallback(ticket_text)
        try:
            prompt_input = {
                "categories": ", ".join(self.categories),
                "ticket_text": ticket_text,
                "format_instructions": self.parser.get_format_instructions(),
            }
            message = self.prompt.invoke(prompt_input)
            with self._generation_observation(
                "llm_classify_ticket",
                input=prompt_input,
                model=getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None),
            ) as observation:
                raw = self.llm.invoke(message)
                observation.update(output=str(getattr(raw, "content", raw)))
            classification = self.parser.parse(str(getattr(raw, "content", raw)))
        except Exception:
            return self._fallback(ticket_text)
        if classification.category not in self.categories:
            return TicketClassification(category=UNKNOWN_CATEGORY, requires_human_review=True)
        return classification

    async def aclassify(self, ticket_text: str) -> TicketClassification:
        if self._is_small_talk(ticket_text):
            return TicketClassification(category=GENERAL_CATEGORY, requires_human_review=False)
        if self.llm is None:
            return self._fallback(ticket_text)
        try:
            prompt_input = {
                "categories": ", ".join(self.categories),
                "ticket_text": ticket_text,
                "format_instructions": self.parser.get_format_instructions(),
            }
            message = self.prompt.invoke(prompt_input)
            with self._generation_observation(
                "llm_classify_ticket",
                input=prompt_input,
                model=getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None),
            ) as observation:
                raw = await _call_llm_async(self.llm, message)
                observation.update(output=str(getattr(raw, "content", raw)))
            classification = self.parser.parse(str(getattr(raw, "content", raw)))
        except Exception:
            return self._fallback(ticket_text)
        if classification.category not in self.categories:
            return TicketClassification(category=UNKNOWN_CATEGORY, requires_human_review=True)
        return classification

    def _fallback(self, ticket_text: str) -> TicketClassification:
        if self._is_small_talk(ticket_text):
            return TicketClassification(category=GENERAL_CATEGORY, requires_human_review=False)
        category = UNKNOWN_CATEGORY
        for candidate in self.categories:
            if candidate not in {GENERAL_CATEGORY, UNKNOWN_CATEGORY} and candidate in ticket_text:
                category = candidate
                break
        requires_human_review = category == UNKNOWN_CATEGORY or any(keyword in ticket_text for keyword in REVIEW_KEYWORDS)
        return TicketClassification(category=category, requires_human_review=requires_human_review)

    def _is_small_talk(self, ticket_text: str) -> bool:
        normalized = ticket_text.strip().lower()
        if not normalized:
            return False
        return any(pattern in normalized for pattern in SMALL_TALK_PATTERNS) and len(normalized) <= 80

    def _generation_observation(self, name: str, **kwargs: object):
        if self.tracer is None:
            from contextlib import nullcontext

            return nullcontext(type("NoopObservation", (), {"update": lambda self, **_kwargs: None})())
        return self.tracer.observation(
            name,
            as_type="generation",
            model_parameters={"temperature": 0},
            **kwargs,
        )


class KnowledgeRetriever:
    def __init__(self, settings: Settings, top_k: int = 3, tracer: LangfuseTracer | None = None) -> None:
        self.settings = settings
        self.top_k = top_k
        self.tracer = tracer

    def retrieve(self, category: str, ticket_text: str) -> list[RetrievedContext]:
        query = f"{category}\n{ticket_text}"
        contexts: list[RetrievedContext] = []
        with self._retriever_observation(
            "chroma_retrieve_context",
            input={"query": query, "k": self.top_k, "collection": self.settings.chroma_collection},
        ) as observation:
            raw_contexts = retrieve_context(self.settings, query, k=self.top_k)
            observation.update(
                output=[
                    {
                        "domain": context.get("domain"),
                        "source": context.get("source"),
                        "score": context.get("score"),
                        "characters": len(str(context.get("text", ""))),
                    }
                    for context in raw_contexts
                ],
                metadata={"returned": len(raw_contexts)},
            )
        for context in raw_contexts:
            contexts.append(
                RetrievedContext(
                    domain=str(context.get("domain", "unknown")),
                    source=str(context.get("source", "unknown")),
                    text=str(context.get("text", "")),
                    score=float(str(context.get("score", 0.0))),
                )
            )
        return contexts

    async def aretrieve(self, category: str, ticket_text: str) -> list[RetrievedContext]:
        query = f"{category}\n{ticket_text}"
        contexts: list[RetrievedContext] = []
        with self._retriever_observation(
            "chroma_retrieve_context",
            input={"query": query, "k": self.top_k, "collection": self.settings.chroma_collection},
        ) as observation:
            raw_contexts = await _retrieve_context_compatible(self.settings, query, self.top_k)
            observation.update(
                output=[
                    {
                        "domain": context.get("domain"),
                        "source": context.get("source"),
                        "score": context.get("score"),
                        "characters": len(str(context.get("text", ""))),
                    }
                    for context in raw_contexts
                ],
                metadata={"returned": len(raw_contexts)},
            )
        for context in raw_contexts:
            contexts.append(
                RetrievedContext(
                    domain=str(context.get("domain", "unknown")),
                    source=str(context.get("source", "unknown")),
                    text=str(context.get("text", "")),
                    score=float(str(context.get("score", 0.0))),
                )
            )
        return contexts

    def _retriever_observation(self, name: str, **kwargs: object):
        if self.tracer is None:
            from contextlib import nullcontext

            return nullcontext(type("NoopObservation", (), {"update": lambda self, **_kwargs: None})())
        return self.tracer.observation(name, as_type="retriever", **kwargs)


class AnswerGenerator:
    def __init__(self, llm: Any, tracer: LangfuseTracer | None = None) -> None:
        self.llm = llm
        self.tracer = tracer
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты пишешь короткие ответы поддержки на русском языке для PoC. "
                    "Используй только предоставленный контекст базы знаний. "
                    "Если контекста недостаточно, скажи, что тикет нужно передать оператору. "
                    "Не запрашивай полный номер карты, пароль, CVC/CVV, токены или коды 2FA.",
                ),
                (
                    "human",
                    "Категория: {category}\n"
                    "Тикет: {ticket_text}\n\n"
                    "Контекст базы знаний:\n{context}\n\n"
                    "Подготовь ответ:",
                ),
            ]
        )

    def generate(self, category: str, ticket_text: str, contexts: list[RetrievedContext]) -> str:
        if category == GENERAL_CATEGORY and not contexts:
            return "Здравствуйте! Напишите, пожалуйста, с чем нужна помощь, и я постараюсь подсказать."
        if not contexts:
            return "Недостаточно данных в базе знаний. Передайте тикет оператору для ручной проверки."
        context_text = self._format_context(contexts)
        if self.llm is None:
            return self._fallback_answer(category, contexts)
        try:
            prompt_input = {
                "category": category,
                "ticket_text": ticket_text,
                "context": context_text,
            }
            message = self.prompt.invoke(prompt_input)
            with self._generation_observation(
                "llm_generate_answer",
                input=prompt_input,
                model=getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None),
            ) as observation:
                raw = self.llm.invoke(message)
                observation.update(output=str(getattr(raw, "content", raw)))
            answer = str(getattr(raw, "content", raw)).strip()
        except Exception:
            return self._fallback_answer(category, contexts)
        return answer or self._fallback_answer(category, contexts)

    async def agenerate(self, category: str, ticket_text: str, contexts: list[RetrievedContext]) -> str:
        if category == GENERAL_CATEGORY and not contexts:
            return "Здравствуйте! Напишите, пожалуйста, с чем нужна помощь, и я постараюсь подсказать."
        if not contexts:
            return "Недостаточно данных в базе знаний. Передайте тикет оператору для ручной проверки."
        context_text = self._format_context(contexts)
        if self.llm is None:
            return self._fallback_answer(category, contexts)
        try:
            prompt_input = {
                "category": category,
                "ticket_text": ticket_text,
                "context": context_text,
            }
            message = self.prompt.invoke(prompt_input)
            with self._generation_observation(
                "llm_generate_answer",
                input=prompt_input,
                model=getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None),
            ) as observation:
                raw = await _call_llm_async(self.llm, message)
                observation.update(output=str(getattr(raw, "content", raw)))
            answer = str(getattr(raw, "content", raw)).strip()
        except Exception:
            return self._fallback_answer(category, contexts)
        return answer or self._fallback_answer(category, contexts)

    def _format_context(self, contexts: list[RetrievedContext]) -> str:
        return "\n\n".join(
            f"Источник: {context.source}\nДомен: {context.domain}\nТекст: {context.text}" for context in contexts
        )

    def _fallback_answer(self, category: str, contexts: list[RetrievedContext]) -> str:
        first = contexts[0]
        return f"Черновик ответа по категории '{category}'. Используйте источник {first.source}: {first.text[:500]}"

    def _generation_observation(self, name: str, **kwargs: object):
        if self.tracer is None:
            from contextlib import nullcontext

            return nullcontext(type("NoopObservation", (), {"update": lambda self, **_kwargs: None})())
        return self.tracer.observation(
            name,
            as_type="generation",
            model_parameters={"temperature": 0},
            **kwargs,
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
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.tracer = tracer
        llm = DeepSeekChatFactory(settings).create()
        categories = self._load_categories(settings.knowledge_dir)
        self.preprocessor = preprocessor or TicketTextPreprocessor()
        self.classifier = classifier or TicketClassifier(llm=llm, categories=categories, tracer=tracer)
        self.retriever = retriever or KnowledgeRetriever(settings=settings, tracer=tracer)
        self.answer_generator = answer_generator or AnswerGenerator(llm=llm, tracer=tracer)

    def process(self, text: str, channel: str, user_id: str | None) -> dict[str, object]:
        with self._span("preprocess_ticket", input={"characters": len(text)}) as observation:
            normalized_text, redacted_text = self.preprocessor.prepare(text)
            observation.update(
                output={
                    "normalized_text": normalized_text,
                    "redacted_text": redacted_text,
                    "redacted": normalized_text != redacted_text,
                }
            )
        classification = self.classifier.classify(redacted_text)
        contexts = self.retriever.retrieve(classification.category, redacted_text)
        requires_human_review = classification.requires_human_review or (
            not contexts and classification.category != GENERAL_CATEGORY
        )
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
        with self._span(
            "persist_ticket_decision",
            input={"ticket_id": result.ticket_id, "decision": decision, "requires_human_review": requires_human_review},
        ) as observation:
            self.store.append_audit({"event": "processed", "channel": channel, "user_id": user_id, **payload})
            if requires_human_review:
                self.store.append_pending(payload)
            observation.update(output={"sources": sources, "category": classification.category})
        return payload

    async def aprocess(self, text: str, channel: str, user_id: str | None) -> dict[str, object]:
        with self._span("preprocess_ticket", input={"characters": len(text)}) as observation:
            normalized_text, redacted_text = self.preprocessor.prepare(text)
            observation.update(
                output={
                    "normalized_text": normalized_text,
                    "redacted_text": redacted_text,
                    "redacted": normalized_text != redacted_text,
                }
            )
        classification = await self._call_component(self.classifier, "aclassify", "classify", redacted_text)
        contexts = await self._call_component(self.retriever, "aretrieve", "retrieve", classification.category, redacted_text)
        requires_human_review = classification.requires_human_review or (
            not contexts and classification.category != GENERAL_CATEGORY
        )
        answer = await self._call_component(
            self.answer_generator,
            "agenerate",
            "generate",
            classification.category,
            redacted_text,
            contexts,
        )
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
        with self._span(
            "persist_ticket_decision",
            input={"ticket_id": result.ticket_id, "decision": decision, "requires_human_review": requires_human_review},
        ) as observation:
            await self.store.append_audit_async({"event": "processed", "channel": channel, "user_id": user_id, **payload})
            if requires_human_review:
                await self.store.append_pending_async(payload)
            observation.update(output={"sources": sources, "category": classification.category})
        return payload

    async def _call_component(self, component: object, async_name: str, sync_name: str, *args: object) -> Any:
        async_method = getattr(component, async_name, None)
        if async_method is not None:
            result = async_method(*args)
            if inspect.isawaitable(result):
                return await result
            return result
        return await asyncio.to_thread(getattr(component, sync_name), *args)

    def _span(self, name: str, **kwargs: object):
        if self.tracer is None:
            from contextlib import nullcontext

            return nullcontext(type("NoopObservation", (), {"update": lambda self, **_kwargs: None})())
        return self.tracer.observation(name, as_type="span", **kwargs)

    def _load_categories(self, knowledge_dir: Path) -> list[str]:
        categories = sorted(path.stem for path in knowledge_dir.glob("*.txt"))
        if GENERAL_CATEGORY not in categories:
            categories.append(GENERAL_CATEGORY)
        if UNKNOWN_CATEGORY not in categories:
            categories.append(UNKNOWN_CATEGORY)
        return categories or list(DEFAULT_CATEGORIES)


def process_ticket(
    settings: Settings,
    store: JsonlStore,
    text: str,
    channel: str,
    user_id: str | None,
    tracer: LangfuseTracer | None = None,
) -> dict[str, object]:
    return TicketPipeline(settings=settings, store=store, tracer=tracer).process(
        text=text,
        channel=channel,
        user_id=user_id,
    )


async def process_ticket_async(
    settings: Settings,
    store: JsonlStore,
    text: str,
    channel: str,
    user_id: str | None,
    tracer: LangfuseTracer | None = None,
) -> dict[str, object]:
    return await TicketPipeline(settings=settings, store=store, tracer=tracer).aprocess(
        text=text,
        channel=channel,
        user_id=user_id,
    )
