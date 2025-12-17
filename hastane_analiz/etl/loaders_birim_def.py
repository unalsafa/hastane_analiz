from typing import Iterable
import pandas as pd
from hastane_analiz.db.connection import batch_insert

def upsert_birim_def(df: pd.DataFrame) -> None:
    """
    DataFrame'deki satirlari hastane_analiz.birim_def tablosuna UPSERT eder.
    ON CONFLICT (kurum_kodu) DO UPDATE...
    """

    if df.empty:
        print("[BIRIM_DEF] Gonderilecek satir yok.")
        return

    sql = """
        INSERT INTO hastane_analiz.birim_def (
            kurum_kodu,
            birim_adi,
            ilce_adi,
            baskanlik_adi,
            kurum_rol_adi,
            kurum_tipi,
            aktif_mi,
            son_guncelleme
        )
        VALUES (%s, %s, %s, %s, %s, %s, TRUE, now())
        ON CONFLICT (kurum_kodu) DO UPDATE SET
            birim_adi      = EXCLUDED.birim_adi,
            ilce_adi       = EXCLUDED.ilce_adi,
            baskanlik_adi  = EXCLUDED.baskanlik_adi,
            kurum_rol_adi  = EXCLUDED.kurum_rol_adi,
            kurum_tipi     = EXCLUDED.kurum_tipi,
            aktif_mi       = TRUE,
            son_guncelleme = now();
    """

    rows: list[tuple] = []
    for _, row in df.iterrows():
        rows.append(
            (
                int(row["kurum_kodu"]),
                row.get("birim_adi"),
                row.get("ilce_adi"),
                row.get("baskanlik_adi"),
                row.get("kurum_rol_adi"),
                row.get("kurum_tipi")
            )
        )

    batch_insert(sql, rows)
    print(f"[BIRIM_DEF] Upsert edilen satir sayisi: {len(rows)}")
