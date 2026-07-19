# Self Review

Самая слабая часть решения - качество retrieval: deterministic hash embedding подходит для PoC, но не заменяет нормальные multilingual sentence embeddings. Классификация также сделана правилами, поэтому она прозрачна, но плохо масштабируется на реальные формулировки пользователей.

Главное предположение: для демонстрации ценности достаточно одного safe path и одного risky path. Нерешенные риски: PII-redaction перед LLM, защита от prompt injection, настоящая очередь, SLA при пиках и качество разметки исторических тикетов.

За два дополнительных дня я бы добавил нормальные embeddings, Docker healthchecks для frontend/ml-service, интеграционный тест compose и реальный DeepSeek-вызов с redaction. Перед production нужны Postgres/Event Log, очередь, rate limits, мониторинг стоимости LLM, A/B или shadow-пилот.

Полностью автоматизировать не стоит legal, refund, account recovery, security incidents и обращения с персональными данными. Проект нужно остановить, если пилот показывает рост reopen rate выше guardrail, ухудшение CSAT или частые небезопасные draft-ответы в risky категориях.
