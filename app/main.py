import streamlit as st

from core.chains import build_chain, sources_line
from infra.chroma_db import get_vector_store
from infra.db import get_engine
from infra.history import SQLiteChatHistory
from infra.paperless import PaperlessError
from infra.paperless_sync import SyncError, sync_paperless
from infra.settings import settings

USER_AVATAR = "👨🏽‍💻"
ASSISTANT_AVATAR = "🥷🏾"

st.set_page_config(page_title="Paperless-Chat", page_icon="💬")


@st.cache_resource
def get_chain():
    return build_chain()


try:
    chain = get_chain()
except Exception as e:
    st.error(f"Start fehlgeschlagen. Ist ChromaDB erreichbar ({settings.CHROMA_HOST}:{settings.CHROMA_PORT})?\n\n{e}")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

history = SQLiteChatHistory(st.session_state.session_id, get_engine())

st.title("💬 Paperless-Chat")
st.caption("Stell Fragen zu deinen Dokumenten aus Paperless-ngx.")

if st.button("Neues Gespräch"):
    history.clear()
    st.rerun()

# Bisherigen Verlauf anzeigen
for message, created_at in history.timestamped_messages():
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role, avatar=USER_AVATAR if role == "user" else ASSISTANT_AVATAR):
        # Quellen stehen getrennt von der Antwort (ältere Einträge haben sie noch im Text)
        st.markdown(message.content + sources_line(message.additional_kwargs.get("sources", [])))
        st.caption(created_at.strftime("%H:%M:%S"))

# Neue Frage
if question := st.chat_input("Ich höre :D"):
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(question)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.write_stream(chain.stream(question, st.session_state.session_id))

# Sidebar: Paperless-Abgleich
with st.sidebar:
    st.header("🗂️ Dokumente")

    try:
        st.metric("Indexierte Textabschnitte", get_vector_store()._collection.count())
    except Exception:
        st.caption("Anzahl nicht abrufbar.")

    if not settings.paperless_configured():
        st.warning("PAPERLESS_URL und PAPERLESS_TOKEN sind nicht gesetzt.")
    else:
        st.caption(
            f"Quelle: {settings.PAPERLESS_URL}. Der Sync-Dienst gleicht automatisch alle "
            f"{settings.SYNC_INTERVAL_MIN} Minuten ab."
        )
        full = st.checkbox("Alles neu einlesen", help="Nötig nach Änderungen am Embedding-Modell oder bei Problemen.")
        if st.button("Jetzt mit Paperless abgleichen", type="primary"):
            bar = st.progress(0, text="Starte Abgleich …")

            def on_progress(done: int, total: int, title: str):
                bar.progress(done / total, text=f"{done}/{total}: {title}")

            try:
                result = sync_paperless(get_vector_store(), full=full, progress=on_progress)
                bar.empty()
                st.success(result.summary())
            except (PaperlessError, SyncError) as e:
                bar.empty()
                st.error(str(e))
