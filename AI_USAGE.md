# Использование AI

AI использовался как помощник для понимания задания, декомпозиции требований и проверки минимального PoC-скоупа. С его помощью были выделены обязательные артефакты: `README.md`, `docs/product.md`, `docs/architecture.md`, `docs/ml.md`, `docs/monitoring.md`, `docs/risks-and-ops.md`, Proof-of-Concept, `AI_USAGE.md`, `WORKLOG.md`, `SELF_REVIEW.md`.

## Где AI помог

- Понять ограничения задания: не строить production-ready систему, а показать happy path, fallback/risky path, компромиссы и проверку качества.
- Сформировать архитектурный скелет PoC: FastAPI ML service, Streamlit UI, Chroma retrieval, DeepSeek-compatible `ChatOpenAI`, optional Langfuse tracing.
- Выбрать минимальный ML/LLM-подход: regex PII masking, классификация, retrieval по базе знаний, grounded draft и human review для risky/low-context тикетов.
- Подготовить тестовые сценарии без реального вызова DeepSeek API: fake LLM и deterministic components делают проверки воспроизводимыми.
- Найти риски и ограничения: PII во внешнем LLM, prompt injection, отсутствие очередей, hash embeddings как PoC-заглушка, JSONL вместо production event log.

## Что было отклонено или отложено

- LangGraph не используется: текущий workflow линейный, и дополнительный orchestration усложнил бы PoC без пользы для проверки.
- Отдельная ML-модель не обучалась: в заданный timebox важнее показать связанный end-to-end pipeline и честно описать данные, которые нужны для обучения.
- Очереди, rate limiting, SLA-оркестрация и highload path не реализованы: это target-часть, а не минимальный PoC.
- Production UI оператора не проектировался подробно, потому что задание прямо просит не уходить в детальный интерфейс оператора.
- TODO: требуется решение автора — какие еще предложения AI были отклонены в процессе работы.

## Самостоятельные решения автора

Финальная граница PoC выбрана вокруг проверяемого demo path: reindex базы знаний, safe ticket draft, risky ticket moderation и audit log. Решение оставить risky categories на human review принято как safety-first компромисс. Решение не обещать production-ready и явно вынести очереди/highload/rate limiting в target architecture также является осознанным scope control.

TODO: требуется решение автора — какие ключевые решения автор принял самостоятельно вне уже описанного scope.

## Что изменилось после работы с AI

После проверки с AI документация стала более явно разделять реализованный PoC и целевую архитектуру. В формулировках появились проверяемые статусы `auto_draft_ready` и `needs_human_review`, отдельные guardrails, явные fallback paths и TODO для неподтвержденных фактов.

## Ошибки AI и как они исправлялись

- AI мог предлагать production-компоненты вроде сложного MLOps, orchestration или очередей как будто они уже реализованы. Это исправлялось сверкой с кодом и переносом таких пунктов в раздел о различиях PoC и целевой архитектуры.
- AI мог завышать уверенность в качестве hash embeddings. Это исправлено явной формулировкой: hash embeddings — PoC-заглушка, optional E5 — более реалистичный локальный вариант.
- AI мог подставлять cost assumptions без верификации. Это исправлено TODO: стоимость LLM-инференса не финализируется, потому что текущий Langfuse пишет token usage как `0`, а полный безопасный замер требует отдельного решения автора.
- TODO: требуется решение автора — конкретные примеры ошибок AI из личного процесса разработки и способ их обнаружения.
