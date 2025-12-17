# hastane_analiz/validation/yearly_outliers.py

from __future__ import annotations
from typing import Literal
import psycopg2
import psycopg2.extras

from hastane_analiz.db.connection import get_connection


Severity = Literal["INFO", "WARN", "ERROR"]


def _insert_validation_run(conn, kategori: str, yil: int, status: str, aciklama: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hastane_analiz.validation_run (kategori, yil, status, aciklama)
            VALUES (%s, %s, %s, %s)
            RETURNING run_id;
            """,
            (kategori, yil, status, aciklama),
        )
        (run_id,) = cur.fetchone()
    return run_id


def run_yearly_outlier_scan(kategori: str, yil: int,
                            sapma_esigi: float = 0.5,
                            min_ort: float = 10.0) -> int:
    """
    Verilen kategori + yıl için:
      - Her kurum + metrik için 12 aya bakar
      - Her ayı 'diğer ayların ortalaması'na göre karşılaştırır
      - Çok düşük/yüksek aylar için validation_issue üretir
      - Eksik ayları da işaretler

    Dönen değer: validation_run.run_id
    """
    with get_connection() as conn:
        run_id = _insert_validation_run(
            conn,
            kategori=kategori,
            yil=yil,
            status="RUNNING",
            aciklama=f"{yil} yılı {kategori} sapma taraması",
        )

        # 1) Outlier aylar (leave-one-out ortalama)
        sql_outliers = """
            WITH aylik AS (
                SELECT
                    yil,
                    ay,
                    kurum_kodu,
                    metrik_adi,
                    SUM(aylik_deger) AS aylik_deger
                FROM hastane_analiz.v_metrik_aylik
                WHERE kategori = %s
                  AND yil = %s
                GROUP BY yil, ay, kurum_kodu, metrik_adi
            ),
            istatistik AS (
                SELECT
                    yil,
                    ay,
                    kurum_kodu,
                    metrik_adi,
                    aylik_deger,
                    SUM(aylik_deger) OVER (
                        PARTITION BY kurum_kodu, metrik_adi
                    ) AS toplam_hepsi,
                    COUNT(*) OVER (
                        PARTITION BY kurum_kodu, metrik_adi
                    ) AS ay_sayisi
                FROM aylik
            )
            SELECT
                yil,
                ay,
                kurum_kodu,
                metrik_adi,
                aylik_deger,
                CASE
                    WHEN ay_sayisi > 1 THEN
                        (toplam_hepsi - aylik_deger) / (ay_sayisi - 1)
                    ELSE NULL
                END AS diger_ay_ort,
                CASE
                    WHEN ay_sayisi > 1
                         AND (toplam_hepsi - aylik_deger) / (ay_sayisi - 1) > 0
                    THEN aylik_deger
                         / ((toplam_hepsi - aylik_deger) / (ay_sayisi - 1))
                    ELSE NULL
                END AS oran
            FROM istatistik;
        """

        issues: list[tuple] = []

        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql_outliers, (kategori, yil))
            for row in cur:
                yil_r = row["yil"]
                ay_r = row["ay"]
                kurum_kodu = row["kurum_kodu"]
                metrik_adi = row["metrik_adi"]
                aylik_deger = row["aylik_deger"]
                diger_ay_ort = row["diger_ay_ort"]
                oran = row["oran"]

                if diger_ay_ort is None or diger_ay_ort < min_ort:
                    continue

                sev: Severity | None = None
                msg: str | None = None

                # Örn: ±%50'den fazla sapma
                if oran >= 1 + sapma_esigi:
                    sev = "WARN"
                    msg = (
                        f"Bu ayın değeri ({aylik_deger}) diğer ayların ortalamasına "
                        f"göre çok yüksek (oran={oran:.2f}, ort={diger_ay_ort:.1f})."
                    )
                elif oran <= 1 - sapma_esigi:
                    sev = "WARN"
                    msg = (
                        f"Bu ayın değeri ({aylik_deger}) diğer ayların ortalamasına "
                        f"göre çok düşük (oran={oran:.2f}, ort={diger_ay_ort:.1f})."
                    )

                if sev is not None:
                    issues.append(
                        (
                            run_id,
                            yil_r,
                            ay_r,
                            kurum_kodu,
                            kategori,
                            metrik_adi,
                            sev,
                            "MONTH_OUTLIER",
                            msg,
                            oran,
                            diger_ay_ort,
                        )
                    )

        # 2) Eksik aylar
        sql_missing = """
            WITH aylar AS (
                SELECT generate_series(1, 12) AS ay
            ),
            mevcut AS (
                SELECT DISTINCT
                    ay,
                    kurum_kodu,
                    metrik_adi
                FROM hastane_analiz.v_metrik_aylik
                WHERE kategori = %s
                  AND yil = %s
            )
            SELECT
                a.ay,
                m.kurum_kodu,
                m.metrik_adi
            FROM aylar a
            CROSS JOIN (
                SELECT DISTINCT kurum_kodu, metrik_adi
                FROM mevcut
            ) m
            LEFT JOIN mevcut y
                ON y.ay = a.ay
               AND y.kurum_kodu = m.kurum_kodu
               AND y.metrik_adi = m.metrik_adi
            WHERE y.ay IS NULL;
        """

        with conn.cursor() as cur:
            cur.execute(sql_missing, (kategori, yil))
            for ay_r, kurum_kodu, metrik_adi in cur:
                msg = "Bu kurum + metrik için bu yılda ilgili ayda hiç veri yok."
                issues.append(
                    (
                        run_id,
                        yil,
                        ay_r,
                        kurum_kodu,
                        kategori,
                        metrik_adi,
                        "WARN",
                        "MONTH_MISSING",
                        msg,
                        None,
                        None,
                    )
                )

        # 3) Issue'ları tek seferde insert et
        if issues:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO hastane_analiz.validation_issue (
                        run_id,
                        yil,
                        ay,
                        kurum_kodu,
                        kategori,
                        metrik_adi,
                        severity,
                        rule_code,
                        message,
                        oran,
                        diger_ay_ort,
                        ekstra_json
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NULL);
                    """,
                    issues,
                )

        # 4) Run status güncelle
        final_status = "OK"
        if any(sev == "ERROR" for *_, sev, _, _, _, _ in issues):
            final_status = "ERROR"
        elif any(sev == "WARN" for *_, sev, _, _, _, _ in issues):
            final_status = "WARN"

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE hastane_analiz.validation_run
                SET status = %s
                WHERE run_id = %s;
                """,
                (final_status, run_id),
            )

        conn.commit()

    return run_id
