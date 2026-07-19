# AI-ассистент обработки тикетов поддержки

PoC показывает минимальный AI/ML-пайплайн для поддержки крупного онлайн-сервиса: тикет нормализуется, PII маскируется, категория определяется через DeepSeek, контекст ищется в Chroma по `knowledge/*.txt`, после чего система готовит черновик ответа или отправляет тикет на human review. Решение покрывает один безопасный happy path и один risky/fallback path.

## Запуск

```bash
cp .env.example .env
# заполнить DEEPSEEK-API-KEY; Langfuse credentials опциональны
docker compose up --build
```

После запуска:

- Streamlit UI: `http://localhost:8501`
- FastAPI health: `http://localhost:8080/health`
- Langfuse UI: `http://localhost:3000`

## Демо

1. В Streamlit запустить переиндексацию базы знаний.
2. Отправить безопасный тикет: `Не приходит код для входа в аккаунт, мой email test@example.com`.
3. Проверить результат: `redacted_text` содержит `[EMAIL]`, категория обычно `auth`, решение `auto_draft_ready`, ответ опирается на источники из `knowledge/*.txt`.
4. Отправить рискованный тикет: `Списали деньги дважды, верните деньги`.
5. Проверить результат: решение `needs_human_review`, тикет появляется в pending moderation и не закрывается автоматически.

## Реализовано

- FastAPI ML service, Streamlit UI и Docker Compose.
- LangChain + DeepSeek-compatible `ChatOpenAI`; текущая модель в конфиге: `deepseek-chat`.
- Chroma retrieval по локальным файлам `knowledge/*.txt`.
- Pydantic-схемы для входа, классификации, retrieval-контекста, ответа и модерации.
- Regex-маскирование email, телефонов и card-like номеров до вызова LLM.
- JSONL audit log и pending moderation для risky/low-context тикетов.
- Optional Langfuse tracing; если ключей нет, сервис продолжает работать без трассировки.
- Локальные deterministic hash embeddings; optional E5 embeddings через `uv sync --extra e5`.

## Ограничения

Это не production-ready система. В PoC не реализованы fast path до 500 мс под highload, очереди, rate limiting, SLA-оркестрация, авторизация, полноценный операторский UI, обучение модели и batch-дедупликация инцидентов. В целевой архитектуре синхронная классификация отделяется от асинхронной генерации, JSONL заменяется на устойчивое хранилище событий, а hash embeddings заменяются на проверенные multilingual embeddings.

Risky tickets не отправляются пользователю без оператора: `legal`, спорные `payments`, `unknown`, low-context и небезопасные случаи уходят в human review. Текущий `deepseek-chat` используется как PoC LLM; по официальной документации DeepSeek это compatibility alias к `deepseek-v4-flash`, deprecated `2026-07-24 15:59 UTC`.

## Бизнес-ценность

Система снижает ручную нагрузку за счет черновиков ответов для типовых обращений и маршрутизации рискованных случаев. Оператор быстрее видит тему, источники и безопасный draft, а пользователь получает более быстрый первый ответ по простым вопросам. Бизнес получает проверяемый способ уменьшать стоимость обработки тикетов без автоматического закрытия юридических, платежных и неопределенных обращений. Главный компромисс PoC: безопасность и аудит важнее процента автозакрытия.
