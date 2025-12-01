# hastane_analiz/db/connection.py

import psycopg2
from psycopg2.extras import execute_batch
from contextlib import contextmanager
from hastane_analiz.config.settings import DB_CONFIG

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

def batch_insert(query: str, rows: list[tuple], page_size: int = 1000) -> None:
    """
    Genel amaçlı batch insert fonksiyonu.
    ETL loader'lar bu fonksiyonu kullanacak.
    """
    if not rows:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_batch(cur, query, rows, page_size=page_size)
            conn.commit()
