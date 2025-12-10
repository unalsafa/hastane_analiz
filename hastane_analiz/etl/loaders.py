# hastane_analiz/etl/loaders.py

import pandas as pd
from hastane_analiz.db.connection import batch_insert
from hastane_analiz.etl.schema_registry import get_raw_veri_insert_sql


def insert_long_df_to_raw_veri(
    long_df: pd.DataFrame,
    kategori: str,
    file_path: str,
    sayfa_adi: str | None = None,
):
    """
    long_df beklenen kolonlar:
      - yil
      - ay
      - kurum_kodu
      - metrik_adi
      - metrik_deger

    kategori: ACIL / DOGUM / AMELIYATHANE ...
    sayfa_adi: genelde sheet adi (ACIL gibi) veya None
    """

    if long_df.empty:
        print("[RAW_VERI] Gonderilecek satir yok.")
        return

    sql = get_raw_veri_insert_sql()
    rows: list[tuple] = []

    for _, row in long_df.iterrows():
        yil = int(row["yil"]) if "yil" in row and pd.notna(row["yil"]) else None
        ay = int(row["ay"]) if "ay" in row and pd.notna(row["ay"]) else None
        kurum_kodu = (
            int(row["kurum_kodu"])
            if "kurum_kodu" in row and pd.notna(row["kurum_kodu"])
            else None
        )

        raw_val = None if pd.isna(row["metrik_deger"]) else row["metrik_deger"]
        num_val = None if pd.isna(row["metrik_deger"]) else float(row["metrik_deger"])

        values = (
            yil,
            ay,
            kurum_kodu,
            kategori,
            sayfa_adi or kategori,          # sheet yoksa kategori ile ayni
            row["metrik_adi"],
            None if raw_val is None else str(raw_val),  # metrik_deger_raw
            num_val,                                     # metrik_deger_numeric
            file_path,
        )
        rows.append(values)

    batch_insert(sql, rows)
    print(f"[RAW_VERI] Insert edilen satir sayisi: {len(rows)}")
