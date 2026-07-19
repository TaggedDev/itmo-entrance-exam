# Риски и эксплуатация

## Высокая нагрузка и надежность

- В PoC pipeline линейный и mostly sync по бизнес-логике: preprocess -> classify -> retrieve -> answer -> log. В PoC не реализовано; в целевой архитектуре fast classification/routing отделяется от slow generation.
- Пики 10-20k тикетов за 10 минут требуют очереди, backpressure и rate limiting. В PoC не реализовано; в целевой архитектуре LLM tasks ставятся в очередь, а risky routing остается на быстром пути.
- При недоступности DeepSeek система не должна закрывать тикет автоматически: fallback классифицирует по правилам, ответ ограничен, тикет уходит в human review.
- Chroma/retrieval сбой или пустой context означает `needs_human_review`; система не должна генерировать факты без источника.

## Приватность, безопасность и риск

- Email, phone и card-like номера маскируются regex-правилами до вызова LLM. Перед production нужны тесты на реальные форматы PII и запрет отправки секретов во внешний API.
- Legal, спорные payments, privacy-запросы, account takeover, fraud и `unknown` нельзя закрывать автоматически; owner решения — оператор поддержки.
- Prompt injection из текста тикета не должен менять системные инструкции классификации и генерации; пользовательский текст считается недоверенным input.
- Все автоматические решения пишутся в JSONL audit log. В production JSONL нужно заменить на неизменяемый event log с поиском по ticket_id, версии prompt/model и источникам ответа.
