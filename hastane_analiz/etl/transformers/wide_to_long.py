import pandas as pd
from typing import Optional, List

from hastane_analiz.etl.transformers.transform_utils import (
    DEFAULT_ID_COLS,
    DEFAULT_IGNORE_COLS,
    load_bool_metrics_for_category,
    normalize_bool_value,
    prepare_common_columns,
    split_metric_columns,
)


def transform_wide_to_long(
    df: pd.DataFrame,
    kategori: Optional[str] = None,
    sayfa_adi: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Genel wide->long donusumu (kategori ve sayfa_adi bilgisi ile DB'deki kural
    tablosundan bool/measure ayrimini yapar).

    Cikis kolonlari: yil, ay, kurum_kodu, metrik_adi, metrik_deger
    """
    df, id_cols = prepare_common_columns(
        df,
        id_cols=id_cols or DEFAULT_ID_COLS,
        convert_id_to_numeric=True,
    )

    ignore_cols = DEFAULT_IGNORE_COLS

    # Kurallardan bool metrik listesini cek (eger kategori verilmis ise DB'den al)
    bool_metric_names = set()
    if kategori:
        try:
            bool_metric_names = load_bool_metrics_for_category(
                kategori,
                sayfa_adi=sayfa_adi,
            )
        except Exception:
            # DB erisimi yoksa veya hata varsa bos set ile devam et
            bool_metric_names = set()

    # Aday metrik kolonlari
    candidate_cols = [c for c in df.columns if c not in id_cols and c not in ignore_cols]

    # Kurala gore bool / numeric ayir
    bool_cols, numeric_cols = split_metric_columns(candidate_cols, bool_metric_names)

    # Sayisal metrikler (default: numeric kabul)
    if numeric_cols:
        df_num = df[id_cols + numeric_cols].copy()
        long_num = df_num.melt(
            id_vars=id_cols,
            value_vars=numeric_cols,
            var_name="metrik_adi",
            value_name="metrik_deger",
        )
        long_num["metrik_deger"] = pd.to_numeric(long_num["metrik_deger"], errors="coerce")
        long_num = long_num.dropna(subset=["metrik_deger"])
    else:
        long_num = pd.DataFrame(columns=id_cols + ["metrik_adi", "metrik_deger"])

    # Bool (Var/Yok, vb.) metrikler -> 1 / 0
    if bool_cols:
        df_bool = df[id_cols + bool_cols].copy()
        for c in bool_cols:
            df_bool[c] = df_bool[c].apply(normalize_bool_value)

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
