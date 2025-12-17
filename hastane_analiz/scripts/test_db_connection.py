from hastane_analiz.db.connection import get_connection

def main():
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print("[DB] Baglanti basarili!")
                print("[DB] PostgreSQL surumu:", version)
    except Exception as e:
        print("[DB] Baglanti hatasi:")
        print(e)

if __name__ == "__main__":
    main()
