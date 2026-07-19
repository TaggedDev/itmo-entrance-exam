# Monitoring

Langfuse поднимается self-hosted через Docker Compose на `http://localhost:3000`.
После регистрации пользователь создает project keys и добавляет `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` в `.env`. ML-сервис пишет traces для обработки
тикета и переиндексации базы знаний.

Технические метрики для PoC: latency `/tickets/process`, ошибки ML-сервиса,
доступность Chroma, количество chunks после reindex, доля ответов без найденного
контекста и доля fallback при недоступности LLM.

ML-метрики: распределение категорий, доля `requires_human_review`, retrieval score,
количество источников в ответе и доля категории `unknown`.

Продуктовые guardrails: reopen rate, CSAT, SLA breach rate и доля тикетов, ушедших
в human review. Стоимость LLM отслеживается по числу DeepSeek-вызовов и размеру
prompt/context.
