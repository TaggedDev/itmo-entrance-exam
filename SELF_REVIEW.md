# Self-review

Самая слабая часть решения — инфраструктура и ML-качество: нет RabbitMQ/Redis/production queue, нет Postgres, нет полноценного state management, а текущий API остается простым PoC-сервисом, а не production ML platform.

Embedding-модель слабая для реального retrieval: hash/fallback и базовый Chroma-поиск подходят для демонстрации, но перед production нужны сравнение embedding-моделей, оценка recall@k/precision@k и тесты на исторических тикетах.

Промпты пока недостаточно точные: нет версионирования, prompt registry, A/B-тестов, regression set для prompt changes и анализа качества разных LLM-провайдеров.

Остаются риски чувствительных данных: PII частично редактируется, но нет гарантированного покрытия всех персональных, платежных и security-sensitive данных до отправки во внешнюю LLM.

Latency можно улучшить: отдельно оптимизировать классификацию на hot path до ориентира 500 мс, вынести медленную генерацию ответа асинхронно, сравнить настройки LLM, prompt length, retrieval top_k и caching.

За 2 дополнительных дня реально добавить Redis/RQ или простой background worker, Postgres вместо JSONL, базовые A/B-тесты промптов, набор offline-eval тикетов и сравнение 2–3 LLM/embedding вариантов.

Перед production обязательно нужны authentication/authorization, личный кабинет оператора, роли доступа, audit permissions, rate limiting, retries, dead-letter queue, мониторинг стоимости LLM и алерты по SLA/ошибкам.

Полностью автоматизировать нельзя risky-категории: legal, refund/payment dispute, account takeover, privacy/security incidents и случаи низкой уверенности модели должны идти оператору.

Пилот стоит остановить, если на больших данных не улучшается CSAT относительно 4.2/5, растет reopen rate выше текущих 9%, ухудшается SLA первого ответа 15 минут, экономия на операторах не покрывает стоимость LLM-инференса или система замедляет обработку и ухудшает ключевые бизнес-метрики.