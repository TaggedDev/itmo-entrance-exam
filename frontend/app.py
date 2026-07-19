import os

import requests
import streamlit as st


ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8080")


st.set_page_config(page_title="AI Support Ticket PoC", layout="wide")
st.title("AI Support Ticket PoC")

section = st.sidebar.radio(
    "Раздел",
    ["Написать тикет", "Модерация тикетов", "Проиндексировать базу знаний"],
)


def api_post(path: str, payload: dict | None = None, timeout: int = 20) -> requests.Response:
    return requests.post(f"{ML_SERVICE_URL}{path}", json=payload or {}, timeout=timeout)


def api_get(path: str, timeout: int = 20) -> requests.Response:
    return requests.get(f"{ML_SERVICE_URL}{path}", timeout=timeout)


def mark_ticket_processing() -> None:
    st.session_state.ticket_processing = True


if "ticket_processing" not in st.session_state:
    st.session_state.ticket_processing = False
if "ticket_result" not in st.session_state:
    st.session_state.ticket_result = None
if "ticket_error" not in st.session_state:
    st.session_state.ticket_error = None


if section == "Написать тикет":
    st.subheader("Написать тикет")
    form_disabled = bool(st.session_state.ticket_processing)

    with st.form("ticket_form"):
        text = st.text_area("Текст обращения", height=180, disabled=form_disabled)
        channel = st.selectbox("Канал", ["web", "chat", "email", "mobile"], disabled=form_disabled)
        user_id = st.text_input("User ID", value="1234", disabled=form_disabled)
        submitted = st.form_submit_button(
            "Обработать",
            disabled=form_disabled,
            on_click=mark_ticket_processing,
        )

    status = st.empty()
    if submitted:
        st.session_state.ticket_result = None
        st.session_state.ticket_error = None
        with status.status("Форма обрабатывается...", expanded=False):
            response = api_post(
                "/tickets/process",
                {"text": text, "channel": channel, "user_id": user_id or None},
            )
        if response.ok:
            st.session_state.ticket_result = response.json()
        else:
            st.session_state.ticket_error = response.text
        st.session_state.ticket_processing = False
        st.rerun()

    if st.session_state.ticket_result:
        result = st.session_state.ticket_result
        st.success("Тикет обработан.")
        left, right = st.columns(2)
        with left:
            st.metric("Category", result["category"])
            st.metric("Human review", str(result["requires_human_review"]))
        with right:
            st.write("Answer")
            st.info(result["answer"])
        st.write("Retrieved context")
        st.json(result["retrieved_context"])
        st.write("Sources")
        st.json(result["sources"])
    elif st.session_state.ticket_error:
        st.error(st.session_state.ticket_error)

elif section == "Модерация тикетов":
    st.subheader("Модерация тикетов")
    response = api_get("/tickets/pending")
    if not response.ok:
        st.error(response.text)
    else:
        pending = response.json()
        if not pending:
            st.info("Нет тикетов на модерации.")
        for ticket in pending:
            title = f"{ticket['ticket_id']} · {ticket['category']} · review={ticket['requires_human_review']}"
            with st.expander(title):
                st.write(ticket["original_text"])
                st.info(ticket["answer"])
                note = st.text_input("Комментарий оператора", key=f"note-{ticket['ticket_id']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve", key=f"approve-{ticket['ticket_id']}"):
                        api_post(
                            f"/tickets/{ticket['ticket_id']}/moderate",
                            {"action": "approve", "operator_note": note},
                        )
                        st.rerun()
                with col2:
                    if st.button("Reject", key=f"reject-{ticket['ticket_id']}"):
                        api_post(
                            f"/tickets/{ticket['ticket_id']}/moderate",
                            {"action": "reject", "operator_note": note},
                        )
                        st.rerun()

else:
    st.subheader("Проиндексировать базу знаний")
    health = api_get("/health")
    if health.ok:
        payload = health.json()
        st.caption(
            f"ML service: {payload['status']} · Chroma: "
            f"{payload['chroma_host']}:{payload['chroma_port']} · "
            f"Embeddings: {payload['embedding_model']}"
        )
    else:
        st.warning("ML-сервис пока не отвечает.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Запустить векторизацию", type="primary"):
            with st.spinner(
                "Индексируем документы и строим embeddings. Первый запуск может занять несколько минут."
            ):
                response = api_post("/knowledge/reindex", timeout=300)
            if response.ok:
                result = response.json()
                st.success(
                    f"База знаний проиндексирована: файлов {result['indexed_files']}, "
                    f"фрагментов {result['indexed_chunks']}."
                )
                st.json(result)
            else:
                st.error(response.text)
    with col2:
        if st.button("Показать чанки"):
            response = api_get("/knowledge/inspect?limit=20", timeout=60)
            if response.ok:
                result = response.json()
                st.success(f"В коллекции `{result['collection']}` сейчас {result['count']} чанков.")
                for item in result["items"]:
                    metadata = item["metadata"]
                    label = (
                        f"{metadata.get('source', 'unknown')} · chunk={metadata.get('chunk', '?')} · "
                        f"dim={item['embedding_dimensions']}"
                    )
                    with st.expander(label):
                        st.write(item["text"])
                        st.write("Metadata")
                        st.json(metadata)
                        st.write("Embedding preview")
                        st.json(item["embedding_preview"])
            else:
                st.error(response.text)
