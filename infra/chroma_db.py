import time
import streamlit as st
import chromadb
import fitz
import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from infra.sharepoint_loader import get_sharepoint_documents_split
from infra.settings import settings

BATCH_SIZE = 64
SLEEP_BETWEEN = 0.05     # Kleine Pause, damit Ollama nicht "zugespammt" wird

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,       # Zeichen pro chunk
    chunk_overlap=150,     # Overlap zwischen Chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_text(text: str) -> list[str]:
    return text_splitter.split_text(text)

# ChromaDB Verbindung
@st.cache_resource
def initialize_vector_db():
    client= chromadb.HttpClient(host="localhost", port=8000, ssl=False)
    if client:
        print("Verbindung zu ChromaDB erfolgreich")
    else:
        print("Verbindung fehlgeschlagen")
    return client

@st.cache_resource
def get_vector_store():
    return Chroma(
        client=initialize_vector_db(),
        collection_name="SharePoint-filtered3", # Hier die Collection eingeben, die genutzt werden soll
        embedding_function=OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
        ),
    )

def _batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def _upload_guard_start(key: str = "upload_running") -> bool:
    # verhindert doppelte Starts (z.B. durch mehrfaches Klicken / Reruns)
    if st.session_state.get(key):
        return False
    st.session_state[key] = True
    return True

def _upload_guard_end(key: str = "upload_running"):
    st.session_state[key] = False

# Upload von lokalen Textdateien
def upload_txt_files(folder_path: str = "./texts") -> int:
    if not _upload_guard_start():
        st.warning("Upload läuft bereits – bitte warten.")
        return 0

    try:
        vector_store = get_vector_store()
        if not os.path.exists(folder_path):
            st.error(f"❌ Ordner '{folder_path}' existiert nicht!")
            return 0

        files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        progress = st.progress(0, text="Starte TXT-Upload…")
        uploaded_files = 0

        for fi, filename in enumerate(files, start=1):
            filepath = os.path.join(folder_path, filename)

            # Skip wenn vorhanden (nur schneller Check)
            existing = vector_store.get(where={"source": filename}, limit=1)
            if existing and existing.get("ids"):
                progress.progress(int(fi / max(len(files), 1) * 100),
                                 text=f"⏭️ Überspringe: {filename} (bereits vorhanden)")
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            # Batch-Upload statt alles auf einmal (stabiler unter Last)
            total = len(chunks)
            for bi, batch in enumerate(_batched(list(enumerate(chunks)), BATCH_SIZE), start=0):
                idxs = [idx for idx, _ in batch]
                texts = [t for _, t in batch]

                ids = [f"{filename}::chunk-{i}" for i in idxs]
                metadatas = [{
                    "source": filename,
                    "path": filepath,
                    "chunk_index": i,
                    "chunk_count": total,
                    "type": "txt",
                } for i in idxs]

                vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                time.sleep(SLEEP_BETWEEN)

            uploaded_files += 1
            progress.progress(int(fi / max(len(files), 1) * 100),
                             text=f"✅ Hochgeladen: {filename} ({total} Chunks)")

        return uploaded_files

    finally:
        _upload_guard_end()

# Extrahieren von Text aus PDFs
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Upload von lokalen PDFs
def upload_pdf_files(folder_path: str = "./pdfs") -> int:
    if not _upload_guard_start():
        st.warning("Upload läuft bereits – bitte warten.")
        return 0

    try:
        vector_store = get_vector_store()
        if not os.path.exists(folder_path):
            st.error(f"❌ Ordner '{folder_path}' existiert nicht!")
            return 0

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        progress = st.progress(0, text="Starte PDF-Upload…")
        uploaded_files = 0

        for fi, filename in enumerate(files, start=1):
            filepath = os.path.join(folder_path, filename)

            existing = vector_store.get(where={"source": filename}, limit=1)
            if existing and existing.get("ids"):
                progress.progress(int(fi / max(len(files), 1) * 100),
                                 text=f"⏭️ Überspringe: {filename} (bereits vorhanden)")
                continue

            text = extract_text_from_pdf(filepath)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            total = len(chunks)
            for batch in _batched(list(enumerate(chunks)), BATCH_SIZE):
                idxs = [i for i, _ in batch]
                texts = [t for _, t in batch]

                ids = [f"{filename}::chunk-{i}" for i in idxs]
                metadatas = [{
                    "source": filename,
                    "path": filepath,
                    "chunk_index": i,
                    "chunk_count": total,
                    "type": "pdf",
                } for i in idxs]

                vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                time.sleep(SLEEP_BETWEEN)

            uploaded_files += 1
            progress.progress(int(fi / max(len(files), 1) * 100),
                             text=f"✅ Hochgeladen: {filename} ({total} Chunks)")

        return uploaded_files

    finally:
        _upload_guard_end()

# Upload von SharePoint-Dokumenten
def upload_sharepoint_library(document_library_id: str, folder_path: str | None = None) -> int:
    if not _upload_guard_start():
        st.warning("Upload läuft bereits – bitte warten.")
        return 0

    try:
        vector_store = get_vector_store()
        docs: list[Document] = get_sharepoint_documents_split(document_library_id=document_library_id)
        if not docs:
            st.warning("⚠️ Keine SharePoint-Dokumente gefunden")
            return 0

        # hier ebenfalls batchen statt alles auf einmal
        texts, metadatas, ids = [], [], []
        progress = st.progress(0, text="Starte SharePoint-Upload…")
        total_docs = len(docs)
        uploaded = 0

        for i, d in enumerate(docs, start=1):
            if not d.page_content or not d.page_content.strip():
                continue

            meta = dict(d.metadata or {})
            base_source = meta.get("name") or meta.get("file_name") or meta.get("source") or f"sp_doc_{i}"
            meta.update({"source": base_source, "type": "sharepoint"})

            texts.append(d.page_content)
            metadatas.append(meta)
            ids.append(f"{base_source}::sp-chunk-{i}")

            if len(texts) >= BATCH_SIZE:
                vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                uploaded += len(texts)
                texts, metadatas, ids = [], [], []
                time.sleep(SLEEP_BETWEEN)

            progress.progress(int(i / max(total_docs, 1) * 100),
                             text=f"SharePoint: {i}/{total_docs} verarbeitet")

        if texts:
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            uploaded += len(texts)

        st.success(f"✅ {uploaded} SharePoint-Chunks geladen")
        return uploaded

    finally:
        _upload_guard_end()
