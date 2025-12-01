import psycopg2
from contextlib import contextmanager
from ..config.settings import DB_CONFIG

@contextmanager
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_cursor():
    with get_connection() as conn:
        with conn.cursor() as cur:
            yield cur
            conn.commit()