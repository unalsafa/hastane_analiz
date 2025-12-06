# hastane_analiz/etl/transformers/acil.py

import pandas as pd

from hastane_analiz.db.connection import get_connection


def _load_acil_bool_metrics() -> set[str]:
    """
    acil_kural_def tablosundan veri_tipi = 'bool' olan
    metrik_yolu değerlerini çeker.

    Bu isimler, ACIL Excel'indeki kolon adları (long path) ile
    birebir aynı olmalı. Örn:
      "Acil Müdahale odası var mı?(var ise işaretleyiniz)"
    """
    sql = """
        SELECT DISTINCT metrik_yolu
        FROM hastane_analiz.acil_kural_def
        WHERE kategori = 'ACIL'
          AND veri_tipi = 'bool'
          AND aktif_mi = TRUE
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        return set()

    return set(df["metrik_yolu"].astype(str))


def _normalize_bool_value(val) -> float:
    """
    Var/Yok ve benzeri değerleri 1.0 / 0.0'a çevirir.
    Boş / anlamsız değerler default olarak 0 kabul edilir.
    """
    if val is None:
        return 0.0

    s = str(val).strip().lower()

    # "1" / "var" / "evet" / "yes" / "true" / işaretli durumlar
    if s in ("1", "var", "evet", "yes", "true", "x", "✓", "ok"):
        return 1.0

    # "0" / "yok" / "hayır" / boş değerler
    if s in ("0", "yok", "hayir", "hayır", "no", "", "nan", "none"):
        return 0.0

    # Beklenmeyen başka bir string gelirse: şimdilik 0.0
    # (ileride buradan WARN türetebiliriz)
    return 0.0


def transform_acil(df: pd.DataFrame, sheet_name: str = "ACIL") -> pd.DataFrame:
    """
    ACIL sayfasini long form'a cevirir.

    Cikis kolonlari:
      - yil
      - ay
      - kurum_kodu
      - metrik_adi   (Excel kolon adı / metrik_yolu)
      - metrik_deger (numeric, bool'lar 1/0)
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
        df["kurum_kodu"] = pd.to_numeric(
            df["kurum_kodu"], errors="coerce"
        ).astype("Int64")

    # 3) ID kolonlari: sadece tarih + kurum
    id_cols = ["yil", "ay", "kurum_kodu"]
    for col in id_cols:
        if col not in df.columns:
            df[col] = None

    # 4) Dim / açıklama kolonlarını hariç tut
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

    # 5) Kurallardan bool metrik listesini çek
    bool_metric_names = _load_acil_bool_metrics()

    # Aday metrik kolonları
    candidate_cols = [
        c for c in df.columns if c not in id_cols and c not in ignore_cols
    ]

    # 6) Kurala göre bool / numeric ayır
    bool_cols: list[str] = []
    numeric_cols: list[str] = []

    for c in candidate_cols:
        if c in bool_metric_names:
            bool_cols.append(c)
        else:
            numeric_cols.append(c)

    # 7) Sayisal metrikler (default: numeric kabul)
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

    # 8) Bool (Var/Yok, vb.) metrikler -> 1 / 0
    if bool_cols:
        df_bool = df[id_cols + bool_cols].copy()
        for c in bool_cols:
            df_bool[c] = df_bool[c].apply(_normalize_bool_value)

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
