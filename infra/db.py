import streamlit as st
from sqlalchemy import create_engine

from .settings import settings

@st.cache_resource
def get_engine():
    return create_engine(settings.SQLITE_URL)
