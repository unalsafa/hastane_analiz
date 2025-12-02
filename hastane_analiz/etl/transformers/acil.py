# hastane_analiz/etl/transformers/acil.py

import pandas as pd


def transform_acil(df: pd.DataFrame, sheet_name: str = "ACIL") -> pd.DataFrame:
    """
    ACIL sayfasini long form'a cevirir.

    Cikis kolonlari:
      - yil
      - ay
      - kurum_kodu
      - metrik_adi
      - metrik_deger (numeric, Var/Yok -> 1/0)
    """

    df = df.copy()

    # 1) Kolon isimlerini standart hale getir
    rename_map = {
        "Yil": "yil",
        "Yıl": "yil",
        "Ay": "ay",
        "BirimId": "kurum_kodu",
        "BirimID": "kurum_kodu",
        "Birim Id": "kurum_kodu",
    }
    df.rename(columns=rename_map, inplace=True)

    # 2) yil / ay / kurum_kodu numeric
    if "yil" in df.columns:
        df["yil"] = pd.to_numeric(df["yil"], errors="coerce").astype("Int64")
    if "ay" in df.columns:
        df["ay"] = pd.to_numeric(df["ay"], errors="coerce").astype("Int64")
    if "kurum_kodu" in df.columns:
        df["kurum_kodu"] = pd.to_numeric(df["kurum_kodu"], errors="coerce").astype(
            "Int64"
        )

    # 3) ID kolonlari: sadece tarih + kurum
    id_cols = ["yil", "ay", "kurum_kodu"]
    for col in id_cols:
        if col not in df.columns:
            df[col] = None

    # 4) Metrik adaylari
    ignore_cols = [
        # boyut alanlari:
        "BirimAdi",
        "Birim Adi",
        "IlceAdi",
        "İlce Adi",
        "İlceAdi",
        "BaskanlikAdi",
        "BaşkanlikAdi",
        "BaşkanlıkAdı",
        "KurumRolAdi",
        "Kurum Rol Adi",
        "KurumTipi",
    ]

    candidate_cols = [
        c for c in df.columns if c not in id_cols and c not in ignore_cols
    ]

    numeric_cols: list[str] = []
    bool_cols: list[str] = []

    for c in candidate_cols:
        series = df[c].dropna().astype(str).str.strip()
        if series.empty:
            numeric_cols.append(c)
            continue

        upper_vals = set(series.str.upper().unique())
        # Sadece VAR / YOK görüyorsak -> boolean kolon
        if upper_vals <= {"VAR", "YOK"}:
            bool_cols.append(c)
        else:
            numeric_cols.append(c)

    # 5) Sayisal metrikler
    df_num = df[id_cols + numeric_cols].copy()
    long_num = df_num.melt(
        id_vars=id_cols,
        value_vars=numeric_cols,
        var_name="metrik_adi",
        value_name="metrik_deger",
    )
    long_num["metrik_deger"] = pd.to_numeric(
        long_num["metrik_deger"], errors="coerce"
    )
    long_num = long_num.dropna(subset=["metrik_deger"])

    # 6) Var / Yok metrikleri (1 / 0)
    if bool_cols:
        df_bool = df[id_cols + bool_cols].copy()
        for c in bool_cols:
            df_bool[c] = (
                df_bool[c]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"VAR": 1.0, "YOK": 0.0})
            )

        long_bool = df_bool.melt(
            id_vars=id_cols,
            value_vars=bool_cols,
            var_name="metrik_adi",
            value_name="metrik_deger",
        )
        long_bool = long_bool.dropna(subset=["metrik_deger"])
        long_df = pd.concat([long_num, long_bool], ignore_index=True)
    else:
        long_df = long_num

    return long_df
