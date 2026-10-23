from functools import lru_cache

from sqlalchemy import create_engine

from .settings import settings


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(settings.SQLITE_URL)
