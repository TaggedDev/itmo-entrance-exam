# Мониторинг

Langfuse в PoC поднимается через Docker Compose и пишет traces для reindex и ticket processing, если заданы `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`. Сейчас Langfuse подтверждает latency и payload, но token usage в наблюдениях записывается как `0`, поэтому стоимость LLM требует отдельного замера.

## Метрики

| Группа | Метрика | Зачем нужна | Стартовый порог |
| --- | --- | --- | --- |
| Техническая | latency `/tickets/process` p95 | контролировать горячий путь | PoC: наблюдать; target: classify/routing до 500 мс |
| Техническая | error rate ML service | ловить сбои API | алерт при `> 1%` за 10 минут |
| Техническая | Chroma availability | retrieval не должен молча падать | алерт при недоступности |
| Техническая | доля пустого retrieval | качество индекса и query | алерт при `> 15%` safe тикетов |
| ML | распределение категорий | ловить drift входящего потока | алерт при отклонении `> 30%` от 7-дневного исходного уровня |
| ML | доля `unknown` | видеть неопределенность | алерт при `> 10%` |
| ML | risky false negative | safety routing | `<= 2%` на ручной выборке |
| ML | retrieval score/source match | качество RAG | ручная проверка top sources |
| Продукт | CSAT | guardrail качества | `>= 4.2` |
| Продукт | reopen rate | guardrail неверных ответов | `<= 10%` |
| Продукт | SLA breach rate | исходная бизнес-задача | не хуже исходного уровня, затем снижение |
| Бизнес | cost per processed ticket | контроль экономики LLM | TODO: требуется решение автора |

## Стартовые алерты

| Алерт | Условие | Действие |
| --- | --- | --- |
| LLM недоступен | ошибки DeepSeek `> 5%` за 10 минут | переводить генерацию в fallback, risky в human review |
| Retrieval деградировал | пустой context `> 15%` safe тикетов | проверить Chroma, reindex, качество embeddings |
| Рост `unknown` | `unknown > 10%` или рост `> 30%` к исходному уровню | проверить новые темы и инциденты |
| Safety-риск | risky false negative `> 2%` | остановить автозакрытие, расширить rules/разметку |
| Reopen рост | reopen rate `> 10%` | откатить safe auto-close в suggest mode |
| CSAT падение | CSAT `< 4.2` | остановить rollout и разобрать sample |
| Human review перегруз | human review `> 35%` после suggest | уточнить routing или снизить охват |

## Drift vs model degradation

Drift входящего потока виден как резкий рост новых тем, `unknown`, пустого retrieval и сдвига категорий при стабильной latency и стабильных LLM ошибках. Model degradation виден как рост неверной классификации, unsafe drafts или правок оператора при похожем распределении входа. Для диагностики нужна ручная labeled выборка: если ошибки растут только на новых темах, это drift; если на старых safe темах, это деградация prompt/model/retrieval.

## Стоимость LLM

Стоимость считать по фактическим `prompt_tokens`, `completion_tokens`, числу LLM-вызовов на тикет и тарифу текущей модели. В PoC на один полный safe ticket обычно есть два LLM-вызова: classification и answer generation. `TODO: требуется решение автора`: провести безопасный замер usage без вывода секретов и зафиксировать cost per ticket, потому что текущий Langfuse trace записывает token usage как `0`.

## Проверка бизнес-задачи

Исходная задача решается, если в пилоте одновременно выполняются условия: SLA breach rate не хуже исходного уровня и снижается на safe сегменте, reopen rate `<= 10%`, CSAT `>= 4.2`, risky false negative `<= 2%`, а стоимость LLM не съедает экономию от сокращения ручной обработки.
