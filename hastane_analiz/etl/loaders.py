# hastane_analiz/etl/loaders.py

import pandas as pd
from hastane_analiz.db.connection import batch_insert
from hastane_analiz.etl.schema_registry import RAW_VERI_COLUMNS, get_raw_veri_insert_sql

def insert_long_df_to_raw_veri(
    long_df: pd.DataFrame,
    kategori: str,
    file_path: str,
):
    """
    long_df beklenen kolonlar:
      - yil, ay
      - birim_adi, ilce_adi, baskanlik_adi, kurum_rol_adi
      - metrik_adi, metrik_deger
    """

    sql = get_raw_veri_insert_sql()
    rows: list[list] = []

    for _, row in long_df.iterrows():
        values = [
            int(row["yil"]),
            int(row["ay"]),
            row.get("birim_adi"),
            row.get("ilce_adi"),
            row.get("baskanlik_adi"),
            row.get("kurum_rol_adi"),
            kategori,         # kategori
            kategori,         # sayfa_adi (simdilik kategori ile ayni)
            row["metrik_adi"],
            str(row["metrik_deger"]),    # metrik_deger_raw
            float(row["metrik_deger"]),  # metrik_deger_numeric
            file_path,
        ]
        rows.append(values)

    if rows:
        batch_insert(sql, rows)
