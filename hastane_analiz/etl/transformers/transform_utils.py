"""
Ortak transformer yardimcilari.
Kategori/sayfa bazli bool metrik listesi cekme, bool normalizasyonu ve
genel kolon hazirlama fonksiyonlari burada tutulur.
"""

from typing import Optional, Iterable, List, Tuple, Set

import pandas as pd

from hastane_analiz.db.connection import get_connection

# Standart kolon isimleri ve eslestirme
DEFAULT_ID_COLS: List[str] = ["yil", "ay", "kurum_kodu"]
DEFAULT_RENAME_MAP = {
    "Yil": "yil",
    "Ay": "ay",
    "BirimId": "kurum_kodu",
    "BirimID": "kurum_kodu",
    "Birim Id": "kurum_kodu",
}

# Boyut/aAciklama kolonlari; metrik havuzuna alinmaz
DEFAULT_IGNORE_COLS: List[str] = [
    "BirimAdi",
    "Birim Adi",
    "IlceAdi",
    "Ilce Adi",
    "IlceAdi",
    "BaskanlikAdi",
    "BaskanlikAdi",
    "BaskanlikAdi",
    "KurumRolAdi",
    "Kurum Rol Adi",
    "KurumTipi",
]


def load_bool_metrics_for_category(kategori: str, sayfa_adi: Optional[str] = None) -> Set[str]:
    """
    kural_def tablosundan veri_tipi = 'bool' olan metrik_yolu degerlerini ceker.
    kategori ve sayfa_adi ile filtrelenir.
    """
    sql = """
        SELECT DISTINCT metrik_yolu
        FROM hastane_analiz.acil_kural_def
        WHERE veri_tipi = 'bool'
          AND aktif_mi = TRUE
          AND kategori = %s
    """
    params = (kategori.upper(),)

    if sayfa_adi is not None:
        sql += "\n          AND sayfa_adi = %s"
        params = (kategori.upper(), sayfa_adi)

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        return set()

    return set(df["metrik_yolu"].astype(str))


def normalize_bool_value(val) -> float:
    """
    Var/Yok ve benzeri degerleri 1.0 / 0.0'a cevirir.
    Bos / anlamsiz degerler default olarak 0 kabul edilir.
    """
    if val is None:
        return 0.0

    s = str(val).strip().lower()

    if s in ("1", "var", "evet", "yes", "true", "x", "?", "ok"):
        return 1.0

    if s in ("0", "yok", "hayir", "hayir", "no", "", "nan", "none"):
        return 0.0

    return 0.0


def prepare_common_columns(
    df: pd.DataFrame,
    *,
    rename_map: Optional[dict] = None,
    id_cols: Optional[Iterable[str]] = None,
    convert_id_to_numeric: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Kolon isimlerini normalize eder, eksik id kolonlarini ekler ve gerekirse numeric'e coercek eder.
    """
    rename_map = rename_map or DEFAULT_RENAME_MAP
    id_cols = list(id_cols or DEFAULT_ID_COLS)

    df = df.copy()
    df.rename(columns=rename_map, inplace=True)

    for col in id_cols:
        if col not in df.columns:
            df[col] = None

    if convert_id_to_numeric:
        for col in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df, id_cols


def split_metric_columns(
    candidate_cols: Iterable[str],
    bool_metric_names: Set[str],
) -> Tuple[List[str], List[str]]:
    """
    Bool/numeric ayrimi icin ortak helper.
    """
    bool_cols: List[str] = []
    numeric_cols: List[str] = []

    for c in candidate_cols:
        if c in bool_metric_names:
            bool_cols.append(c)
        else:
            numeric_cols.append(c)

    return bool_cols, numeric_cols
