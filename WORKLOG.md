# Журнал работы

- 15:17-15:45 — подготовка проектного окружения: task/env handling, исключение `.env`, первичная структура под PoC.
- 16:23 — собран первый end-to-end каркас: Streamlit frontend, FastAPI backend/ML service, базовый ML pipeline и `docker compose up`.
- 16:36-16:41 — подготовлена mocked knowledge base, добавлены тесты для embeddings/Chroma и реализовано наполнение Chroma.
- 16:41-16:57 — доработан frontend для запуска индексации базы знаний и обновлена логика embeddings.
- 17:08-17:29 — подключен LangChain-сервис и Langfuse tracing, логирование ML-пайплайна выведено в Langfuse на `:3000`.
- 18:24 — добавлены async-вызовы для Chroma и LLM, сохранена совместимость публичных API.
- 18:37-18:58 — оформлены проектные артефакты: README, Product, Architecture, ML и Monitoring docs.
- 19:03 — финальный документальный слой: добавлен `AI_USAGE.md`, зафиксировано использование AI и итоговое состояние PoC.