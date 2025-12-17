# hastane_analiz/etl/transformers/duzenleyen.py

import pandas as pd
from hastane_analiz.etl.schema_registry import ID_COLS

def transform_duzenleyen(df: pd.DataFrame, yil: int, ay: int) -> pd.DataFrame:
    """
    DUZENLEYEN dosyasini long form'a cevirir.

    Cikis kolonlari:
      - yil, ay
      - birim_adi, ilce_adi, baskanlik_adi, kurum_rol_adi
      - metrik_adi
      - metrik_deger
    """

    df = df.copy()

    # Yil / ay sabit kolon olarak eklensin
    df["yil"] = yil
    df["ay"] = ay

    # Excel'de bu kolonlar yoksa, simdilik bos ekleyelim
    for col in ["birim_adi", "ilce_adi", "baskanlik_adi", "kurum_rol_adi"]:
        if col not in df.columns:
            df[col] = None

    # ID kolonlari sabit
    id_cols = ID_COLS

    # Metrik kolonlari: ID_COLS disinda kalan her sey
    metric_cols = [c for c in df.columns if c not in id_cols]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metrik_adi",
        value_name="metrik_deger",
    )

    # Sayisal olmayanlari NaN yap
    long_df["metrik_deger"] = pd.to_numeric(long_df["metrik_deger"], errors="coerce")

    # metrik_deger bos olanlari at
    long_df = long_df.dropna(subset=["metrik_deger"])

    return long_df
