# 💬 RAG-Chatbot

Ein KI-Assistent auf Basis von **Retrieval-Augmented Generation (RAG)**. Das System ermöglicht die Beantwortung von Fragen auf Basis von Dokumenten aus **Microsoft SharePoint** sowie lokalen Dateien.

## 🚀 Features

* **LLM-Infrastruktur**: Betrieb über **Ollama** für die lokale Verarbeitung.
* **Hybride Datenquellen**:
    * **Lokal**: Einlesen von PDFs und Textdateien aus lokalen Verzeichnissen.
    * **Cloud (Optional)**: Integration mit der Microsoft Graph API zum Einlesen von SharePoint-Dokumenten.
* **Reranking**: Einsatz des `Qwen3-Reranker` Modells zur Gewichtung der Suchergebnisse vor der Antwortgenerierung.
* **Multi-Format Support**: Extraktion von Texten aus **PDF, DOCX, TXT und Markdown**.
* **Historie**: Speicherung der Chat-Verläufe mittels **SQLite** zur Bereitstellung von Kontext in laufenden Sitzungen.

## 🛠️ Technologie-Stack

| Komponente | Technologie |
| :--- | :--- |
| **Orchestrierung** | LangChain & LangChain-Ollama |
| **Vektordatenbank** | ChromaDB (HttpClient Anbindung) |
| **LLM & Embeddings** | Ollama (Llama 3.2, Nomic-Embed-Text) |
| **Web-Interface** | Streamlit |
| **API-Schnittstelle** | Microsoft Graph API (O365) |
| **Datenbank** | SQLAlchemy (SQLite) |

## 📋 Voraussetzungen

1.  **Ollama**: Erreichbarer Ollama-Server mit den Modellen `llama3.2`, `nomic-embed-text-v2-moe` und `dengcao/Qwen3-Reranker-0.6B:Q8_0`.
2.  **Python**: Version 3.10 oder höher.
3.  **ChromaDB**: Laufender ChromaDB-Server.
