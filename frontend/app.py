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


def api_post(path: str, payload: dict | None = None) -> requests.Response:
    return requests.post(f"{ML_SERVICE_URL}{path}", json=payload or {}, timeout=20)


def api_get(path: str) -> requests.Response:
    return requests.get(f"{ML_SERVICE_URL}{path}", timeout=20)


if section == "Написать тикет":
    st.subheader("Написать тикет")
    with st.form("ticket_form"):
        text = st.text_area("Текст обращения", height=180)
        channel = st.selectbox("Канал", ["web", "chat", "email", "mobile"])
        user_id = st.text_input("User ID", value="")
        submitted = st.form_submit_button("Обработать")
    if submitted:
        response = api_post(
            "/tickets/process",
            {"text": text, "channel": channel, "user_id": user_id or None},
        )
        if response.ok:
            result = response.json()
            st.success(f"Decision: {result['decision']}")
            left, right = st.columns(2)
            with left:
                st.metric("Category", result["category"])
                st.metric("Risk", result["risk_level"])
                st.metric("Confidence", result["confidence"])
            with right:
                st.write("Draft response")
                st.info(result["draft_response"])
            st.write("Retrieved context")
            st.json(result["retrieved_context"])
        else:
            st.error(response.text)

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
            with st.expander(f"{ticket['ticket_id']} · {ticket['category']} · {ticket['risk_level']}"):
                st.write(ticket["text"])
                st.info(ticket["draft_response"])
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
        st.json(health.json())
    if st.button("Запустить векторизацию"):
        response = api_post("/knowledge/reindex")
        if response.ok:
            st.success("База знаний проиндексирована.")
            st.json(response.json())
        else:
            st.error(response.text)
