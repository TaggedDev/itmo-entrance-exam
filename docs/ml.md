# ML Design

ML-часть реализована как простой линейный LangChain pipeline:

```text
TicketRequest
-> normalize text
-> redact PII
-> classify with DeepSeek
-> retrieve context from Chroma
-> generate grounded answer with DeepSeek
-> return Pydantic TicketResponse
```

## Preprocessing

Перед вызовом LLM текст приводится к нижнему регистру, лишние пробелы схлопываются,
а чувствительные данные маскируются простыми regex-правилами:

- email -> `[EMAIL]`
- phone -> `[PHONE]`
- card-like number -> `[CARD]`

## Classification

Классификатор использует отдельный prompt и возвращает только Pydantic-схему
`TicketClassification`: `category` и `requires_human_review`.

Список категорий берется из файлов `knowledge/*.txt`: `auth`, `feedback`, `legal`,
`payments`; дополнительно есть fallback `unknown`. Risk level и confidence удалены,
чтобы не усложнять PoC.

## Retrieval And Answering

Retrieval работает через существующую интеграцию LangChain + Chroma. Query строится как
`category + redacted_text`, `top_k=3`. Ответ генерируется DeepSeek только на основе
найденных источников. Если источников нет или LLM недоступна, pipeline деградирует
безопасно: тикет требует human review, а ответ не выдумывает факты.
