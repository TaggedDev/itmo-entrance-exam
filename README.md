# AI Support Ticket Assistant PoC

Минимальный PoC показывает обработку тикетов поддержки: классификация, оценка риска, retrieval из базы знаний Chroma, mock-ответ LLM, запись audit log и отправка рискованных обращений на модерацию.

## Запуск

1. Скопируйте `.env.example` в `.env` и при необходимости заполните `DEEPSEEK-API-KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
2. Запустите стенд:

```bash
docker compose up --build
```

3. Откройте:
   - Streamlit UI: http://localhost:8501
   - Langfuse self-hosted UI: http://localhost:3000
   - ML service health: http://localhost:8080/health

## Demo Scenario

В Streamlit откройте вкладку `Проиндексировать базу знаний` и нажмите кнопку векторизации. Затем во вкладке `Написать тикет` отправьте безопасный запрос: `Не приходит код для входа в аккаунт`. Система вернет категорию `auth`, найденный контекст и draft ответа.

Для risky path отправьте: `У меня юридическая претензия и я пойду в суд`. Система не закрывает тикет автоматически, а кладет его во вкладку `Модерация тикетов`.

## Что реально реализовано

Реализованы Streamlit UI, FastAPI ML service, Chroma indexing/retrieval, JSONL audit log, pending moderation и optional Langfuse tracing. DeepSeek API key читается из `.env` как `DEEPSEEK-API-KEY`, но генерация ответа в MVP остается моковой.

## Упрощения

Нет авторизации, миграций, настоящей очереди, production-хранилища тикетов и реального LLM-вызова. LangGraph не используется, потому что MVP pipeline линейный. В целевой архитектуре JSONL заменяется на Postgres/Event Log, mock LLM - на DeepSeek API, а правила классификации - на ML baseline.
