# ML Design

В MVP есть четыре ML/LLM-задачи: классификация тикета, risk scoring, retrieval из базы знаний и генерация черновика ответа. Классификация и риск сейчас сделаны правилами, потому что это быстрый baseline и он прозрачен для risky категорий.

Retrieval реализован через LangChain + Chroma. База знаний лежит в `knowledge/*.txt`, режется на chunks и индексируется deterministic hash embedding. Такой embedding слабее настоящих sentence embeddings, но не требует скачивания моделей и подходит для PoC.

LLM в MVP не вызывается: ответ генерируется mock-функцией, но ML-сервис сразу читает `DEEPSEEK-API-KEY` из `.env` и имеет стабильную точку для будущего DeepSeek-вызова. Risky категории: legal, refund, account takeover, персональные данные и low-confidence обращения. Они не закрываются автоматически.

LangGraph сознательно отложен. Текущий workflow линейный: classify -> retrieve -> generate draft -> log. Если появятся ретраи, многошаговые состояния, tool calls и ручные переходы между этапами, LangGraph можно добавить без изменения REST-контракта.
