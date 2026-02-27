import streamlit as st
from langchain_ollama import ChatOllama

from infra.settings import settings

@st.cache_resource
def get_llm():
    return ChatOllama(base_url=settings.OLLAMA_BASE_URL,
                      model=settings.OLLAMA_MODEL,
                      temperature=settings.OLLAMA_TEMPERATURE)