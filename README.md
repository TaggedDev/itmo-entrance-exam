# AI Support Ticket Assistant PoC

Минимальный PoC показывает ML pipeline для обработки тикетов поддержки:
нормализация текста, маскирование PII, классификация через DeepSeek, retrieval из Chroma,
генерация ответа на основе базы знаний, audit log и очередь тикетов для human review.

## Запуск

1. Скопируйте `.env.example` в `.env` и заполните `DEEPSEEK-API-KEY`.
   Langfuse credentials опциональны.
2. Запустите стенд:

```bash
docker compose up --build
```

3. Откройте:
   - Streamlit UI: http://localhost:8501
   - Langfuse UI: http://localhost:3000
   - ML service health: http://localhost:8080/health

## Demo Scenario

Сначала во вкладке индексации запустите векторизацию базы знаний. Chroma проиндексирует
файлы из `knowledge/*.txt`; категории тикетов берутся из имен этих файлов.

Затем отправьте тикет:

```text
Не приходит код для входа в аккаунт, мой email test@example.com
```

Pipeline вернет категорию, redacted text, найденные источники и ответ. Для legal/payment
или неуверенных обращений поле `requires_human_review` будет `true`, а тикет попадет в
список модерации.

## Что реализовано

- FastAPI ML service и Streamlit UI.
- LangChain + DeepSeek-compatible `ChatOpenAI`.
- Chroma retrieval поверх уже существующей embedding-модели.
- Pydantic-структуры для входа, классификации, retrieval-контекста и ответа.
- Regex-маскирование email, телефонов и номеров карт.
- JSONL audit log и pending moderation для проверяемых тикетов.

## Упрощения

Это экзаменационный PoC, а не production highload-система. Здесь нет очередей,
rate limiting, авторизации, SLA-оркестрации, обучения модели и сложного branching.
Pipeline намеренно линейный: preprocess -> classify -> retrieve -> answer -> log.
