# Risks And Ops

## Reliability

- В PoC pipeline синхронный и линейный: preprocess -> classify -> retrieve -> answer.
- Очереди, rate limits и highload-оркестрация намеренно не реализованы.
- При недоступности LLM система безопасно деградирует: возвращает ограниченный ответ и отправляет тикет на human review.
- Если Chroma не вернула контекст, ответ не выдумывает факты, а тикет также требует human review.

## Privacy And Safety

- Email, телефоны и card-like номера маскируются до отправки текста в LLM.
- Legal, payment/refund disputes, account takeover и `unknown` требуют human-in-the-loop.
- Prompt injection в пользовательском тексте не должен менять системные инструкции классификации или генерации.
- Все решения пишутся в JSONL audit log, чтобы можно было восстановить причину ответа или эскалации.
