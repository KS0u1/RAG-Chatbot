import os

from dotenv import load_dotenv

load_dotenv()  # lädt Werte aus .env in os.environ

class Settings:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # On-Prem Server
    OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2:1b") # Hier Modell einstellen, welches genutzt werden soll
    OLLAMA_TEMPERATURE  = os.getenv("OLLAMA_TEMPERATURE", "0")
    SQLITE_URL      = os.getenv("SQLITE_URL", "sqlite:///chat_history.db")

    # --- SharePoint / O365 für LangChain ---
    SP_CLIENT_ID = os.getenv("SP_CLIENT_ID")  # Application (client) ID
    SP_CLIENT_SECRET = os.getenv("SP_CLIENT_SECRET")  # Secret Value
    SP_TENANT_ID = os.getenv("SP_TENANT_ID")  # Directory (tenant) ID
    SP_SITE_URL = os.getenv("SP_SITE_URL")
    SP_SITE_ID = os.getenv("SP_SITE_ID")
    SP_DRIVE_ID = os.getenv("SP_DRIVE_ID")

    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
    OLLAMA_RERANK_MODEL = os.getenv("OLLAMA_RERANK_MODEL", "dengcao/Qwen3-Reranker-0.6B:Q8_0")

settings = Settings()
