# hastane_analiz/validation/promote_to_fact.py

from __future__ import annotations

from typing import Optional

from hastane_analiz.db.connection import get_connection


def _build_like_pattern(yil: int, ay: int) -> str:
    """
    Dosya adlarında genelde 2025-01 gibi yıl-ay geçtiği için
    kaynak_dosya LIKE aramasında kullanılacak pattern'i üretir.
    """
    return f"%{yil}-{ay:02d}%"


def has_fatal_issues(kategori: str, yil: int, ay: int) -> bool:
    """
    Verilen kategori + yıl + ay için etl_kalite_sonuc tablosunda
    FATAL seviye issue var mı?

    - Önce direkt yil/ay kolonuna bakar
    - Ek olarak, kaynak_dosya içinde '2025-01' vb. geçen satırları da
      dahil eder (bazı kurallar yıl/ay kolonlarını doldurmasa bile
      dosya adına göre yakalayalım diye).
    """
    kategori_u = kategori.upper()
    like_pattern = _build_like_pattern(yil, ay)

    sql = """
        SELECT EXISTS (
            SELECT 1
            FROM hastane_analiz.etl_kalite_sonuc e
            WHERE e.kategori = %s
              AND e.seviye = 'FATAL'
              AND (
                    (e.yil = %s AND e.ay = %s)
                 OR e.kaynak_dosya LIKE %s
              )
        ) AS has_fatal;
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (kategori_u, yil, ay, like_pattern))
        row = cur.fetchone()
        return bool(row[0])


def get_issue_summary(kategori: str, yil: int, ay: int) -> dict:
    """
    İstersen dashboard tarafında göstermek için
    FATAL / WARN sayılarının özetini döner.
    """
    kategori_u = kategori.upper()
    like_pattern = _build_like_pattern(yil, ay)

    sql = """
        SELECT e.seviye, COUNT(*) AS adet
        FROM hastane_analiz.etl_kalite_sonuc e
        WHERE e.kategori = %s
          AND (
                (e.yil = %s AND e.ay = %s)
             OR e.kaynak_dosya LIKE %s
          )
        GROUP BY e.seviye;
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (kategori_u, yil, ay, like_pattern))
        rows = cur.fetchall()

    out = {"FATAL": 0, "WARN": 0, "INFO": 0}
    for sev, adet in rows:
        out[str(sev)] = int(adet)
    return out


def promote_month_to_fact(
    kategori: str,
    yil: int,
    ay: int,
    *,
    force: bool = False,
) -> None:
    """
    Seçilen kategori + yıl + ay için veriyi fact_metrik_aylik tablosuna yazar.

    - Eğer FATAL issue varsa ve force=False ise RuntimeError fırlatır
      (dashboard tarafında yakalayıp kullanıcıya mesaj gösteriyoruz).
    - force=True verirsen FATAL olsa bile fact'e yazdırabilirsin.
    """

    kategori_u = kategori.upper()

    # 1) Kapı: FATAL var mı?
    if not force and has_fatal_issues(kategori_u, yil, ay):
        raise RuntimeError(
            f"{kategori_u} / {yil}-{ay:02d} için FATAL validation kayıtları bulunduğu "
            "için fact_metrik_aylik tablosuna aktarım yapılmadı."
        )
    
    # 2) Önce bu ayın eski fact kayıtlarını sil
    delete_sql = """
        DELETE FROM hastane_analiz.fact_metrik_aylik
        WHERE kategori = %s
          AND yil = %s
          AND ay = %s;
    """
   
    # 3) v_metrik_aylik (veya raw_veri) üzerinden bu ayın verisini ekle
    # Burada varsayım: v_metrik_aylik şu kolonlara sahip:
    #   yil, ay, kurum_kodu, kategori, sayfa_adi, metrik_adi, metrik_deger
    insert_sql = """
        INSERT INTO hastane_analiz.fact_metrik_aylik (
            yil,
            ay,
            kurum_kodu,
            kategori,
            sayfa_adi,
            metrik_adi,
            metrik_deger
        )
        SELECT
            yil,
            ay,
            kurum_kodu,
            kategori,
            sayfa_adi,
            metrik_adi,
            aylik_deger
        FROM hastane_analiz.v_metrik_aylik
        WHERE kategori = %s
          AND yil = %s
          AND ay = %s;
    """
    print("---- DELETE SQL ----")
    print(delete_sql)
    print("params:", (kategori_u, yil, ay))

    print("---- INSERT SQL ----")
    print(insert_sql)
    print("params:", (kategori_u, yil, ay))
    with get_connection() as conn, conn.cursor() as cur:
        # Eski kayıtları sil
        cur.execute(delete_sql, (kategori_u, yil, ay))

        # Yeni verileri ekle
        cur.execute(insert_sql, (kategori_u, yil, ay))

        conn.commit()

        
    