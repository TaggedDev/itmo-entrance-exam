# Самопроверка

## Самая слабая часть

Самая слабая часть решения — качество retrieval и классификации на реальных пользовательских формулировках. Deterministic hash embeddings подходят для PoC и offline smoke checks, но не заменяют полноценные multilingual embeddings. Классификация через LLM и fallback rules требует ручной labeled проверки перед любым автозакрытием.

## Assumptions

- Одного safe path и одного risky/fallback path достаточно, чтобы показать архитектурную идею.
- Доля типовых обращений равна 40%, стоимость ручной обработки — 150 ₽/тикет, SLA первого ответа — 15 минут, CSAT — 4.2, reopen rate — 9%; это вводные из `task.txt`.
- MVP может начать с 15% safe automation только после shadow/suggest проверки.
- DeepSeek используется как текущий PoC LLM; стоимость LLM-инференса не зафиксирована и требует отдельного замера.

## Нерешенные риски

- PII masking на regex может пропустить реальные форматы документов, адресов или нестандартных номеров.
- Prompt injection проверен только на уровне prompt-дизайна, а не отдельным safety test suite.
- JSONL audit log прозрачен для PoC, но не подходит для production-аудита при высокой нагрузке.
- Нет production queue, rate limiting, backpressure и отдельного fast path до 500 мс.
- Langfuse сейчас пишет token usage как `0`, поэтому cost monitoring не закрыт.

## Что улучшить за 2 дня

- Заменить hash embeddings на E5 и собрать labeled retrieval set.
- Добавить негативные safety-тесты на prompt injection и PII leakage.
- Добавить compose smoke-test для frontend, ML service, Chroma и Langfuse.
- Снять проверяемую стоимость LLM-инференса и заполнить TODO в product/monitoring.
- Расширить тесты на `unknown`, empty context, legal/payments и moderation approve/reject.

## Перед production

- Заменить JSONL на event log/Postgres или аналогичное устойчивое audit-хранилище.
- Вынести fast classification/routing отдельно от slow LLM generation.
- Добавить очередь, rate limiting, auth, secrets management и SLA monitoring.
- Провести shadow/pilot с ручной разметкой и оценкой risky false negative.
- Завести контроль prompt/model versions и rollback-процедуру.

## Что не автоматизировать полностью

Не стоит полностью автоматизировать legal, спорные payments/refund, account takeover, fraud, privacy-запросы, обращения с документами и случаи без надежного retrieved context. Эти категории должны оставаться в human review даже при хорошем draft.

## Критерии остановки пилота

Проект нужно остановить или откатить в suggest mode, если CSAT падает ниже 4.2, reopen rate растет выше 10%, risky false negative выше 2%, unsafe draft появляется в критичных категориях или стоимость LLM-инференса съедает экономию от сокращения ручной обработки.
