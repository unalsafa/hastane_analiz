import pandas as pd

from hastane_analiz.db.connection import get_connection

def _load_dogum_bool_metrics () -> set[str]:
    sql = """ 
        SELECT DISTINTC metrik_yolu
        FROM hastane_analiz.acil_kural_def
        WHERE kategori = 'DOGUM'
        AND veri_tipi = 'bool'
        AND aktif_mi = TRUE
        """

    with get_connection() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        return set()

    return set(df["metrik_yolu"].astype(str))

def _normalize_bool_metrics(val) -> set[str]:
    if val is None:
        return 0.0
    s = str(val).strip().lower()

    if s in ("1", "var", "evet", "yes", "true", "x", "✓", "ok"):
        return 1.0
    if s in ("0", "yok", "hayir", "hayır", "no", "", "nan", "none"):
        return 0.0

    return 0.0
