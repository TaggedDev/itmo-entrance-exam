# Monitoring

Langfuse поднимается self-hosted через Docker Compose на `http://localhost:3000`. После регистрации пользователь создает project keys и добавляет `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` в `.env`. ML-сервис пишет traces для обработки тикета и переиндексации базы знаний.

Технические метрики: latency `/tickets/process`, ошибки ML-сервиса, доступность Chroma, количество chunks после reindex, доля fallback без контекста. ML-метрики: распределение категорий, risk level, confidence, retrieval score, доля low-confidence.

Продуктовые guardrails: reopen rate, CSAT, SLA breach rate и доля тикетов, ушедших в human review. Стоимость LLM отслеживается по числу future DeepSeek-вызовов, токенам и категории тикета; risky/low-confidence обращения не должны тратить LLM на пользовательский auto-response.

Деградация модели отличается от изменения потока через сравнение входного распределения категорий и retrieval score. Если внезапно растет доля legal/payment или unknown, это может быть изменение потока; если при похожем потоке падает confidence/retrieval score, это деградация качества.
