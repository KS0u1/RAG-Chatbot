import streamlit as st

from langchain_community.chat_message_histories import SQLChatMessageHistory
from datetime import datetime
from sqlalchemy import text

from core.chains import build_chain
from infra.db import get_engine
from infra.settings import settings

from infra.chroma_db import (
    upload_pdf_files,
    upload_txt_files,
    get_vector_store,
    upload_sharepoint_library
)

r = build_chain()

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

st.logo("./images/....", size="large")
st.title("💬 RAG-Chatbot")
st.caption("Dieser Chatbot soll dir dabei helfen, dich bei deinem Arbeitsalltag zu assistieren!")

# "Neues Gespräch" Button
if st.button("Neues Gespräch"):
    try:
        SQLChatMessageHistory(
            session_id=st.session_state.session_id,
            connection=get_engine()
        ).clear()
        st.success("Chatverlauf gelöscht.")
    except Exception as e:
        st.warning(f"Verlauf konnte nicht gelöscht werden: {e}")
    st.rerun()

# Nachrichten  aus DB laden mit created_at
with get_engine().connect() as conn:
    result = conn.execute(
        text("SELECT message, created_at FROM message_store WHERE session_id = :sid ORDER BY id"),
        {"sid": st.session_state.session_id}
    ).fetchall()

user_avatar = "👨🏽‍💻"
assistant_avatar = "🥷🏾"

for row in result:
    import json

    msg = json.loads(row[0])
    role = "user" if msg["type"] == "human" else "assistant"
    avatar = user_avatar if role == "user" else assistant_avatar

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["data"]["content"])
        if row[1]:  # created_at liegt als String vor
            # String in datetime umwandeln
            created_at_dt = datetime.fromisoformat(row[1])
            st.caption(f"{created_at_dt.strftime('%H:%M:%S')}")


# Chat-Eingabe
if user := st.chat_input("Ich höre :D"):
    with st.chat_message("user", avatar=user_avatar):
        st.write(user)
        st.caption(f"{datetime.now().strftime('%H:%M:%S')}")

    with st.chat_message("assistant", avatar=assistant_avatar):
        text = st.write_stream(
            r.stream(
                {"input": user},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
        )
        st.caption(f"{datetime.now().strftime('%H:%M:%S')}")

# Sidebar für die Upload-Funktionen
with st.sidebar:
    st.header("🗂️ Dokumente verwalten")
    if st.button("📁 TXT-Dateien hochladen", type="primary"):
        uploaded = upload_txt_files("./texts")  # Ordner mit Testdateien
    if st.button("📄 PDF-Dateien hochladen", type="primary"):
        uploaded_pdfs = upload_pdf_files("./pdfs")
    if st.button("☁️ SharePoint-Dokumente laden", type="primary"):
        st.write("Upload läuft… (bitte Tab offen lassen)")
        count = upload_sharepoint_library(settings.SP_DRIVE_ID)
        st.success(f"{count} SharePoint-Dokumente in Chroma geladen.")