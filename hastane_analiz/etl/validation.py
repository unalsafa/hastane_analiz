from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Callable

import json
import pandas as pd
import numpy as np

from hastane_analiz.db.connection import get_connection, batch_insert


class Severity(str, Enum):
    FATAL = "FATAL"
    WARN = "WARN"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    severity: Severity
    rule_code: str
    message: str
    file_path: Optional[str] = None
    kategori: Optional[str] = None
    sayfa_adi: Optional[str] = None
    row_index: Optional[int] = None
    context: Optional[Dict[str, Any]] = None

# ==========================================================
#  GENEL HEURISTIC KURALLAR (KATEGORİDEN BAĞIMSIZ)
# ==========================================================

def v_zero_while_others_positive_generic(
    df: pd.DataFrame,
    group_cols: list[str],
    id_col: str,
    metric_col: str = "metrik_deger",
    min_group_size: int = 5,
    min_positive_ratio: float = 0.7,
    rule_code: str = "ZERO_WHILE_OTHERS_POSITIVE",
    severity: Severity = Severity.WARN,
) -> List[ValidationIssue]:
    """
    Aynı group_cols grubu içinde kurumların çoğu > 0 iken
    0 veya NaN olan satırlar için WARN/FATAL üretir.

    Örn: group_cols = ["yil","ay","metrik_adi"], id_col="kurum_kodu"
    """
    issues: List[ValidationIssue] = []

    required = set(group_cols + [id_col, metric_col])
    if not required.issubset(df.columns):
        return issues

    work = df.copy()
    for gc in group_cols:
        work[gc] = pd.to_numeric(work[gc], errors="ignore")

    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")

    grp = work.groupby(group_cols, dropna=False)

    for key, sub in grp:
        if len(sub) < min_group_size:
            continue

        total = len(sub)
        positive = (sub[metric_col] > 0).sum()

        # Çoğunluk zaten 0/NaN ise anlamlı değil
        if total == 0 or positive / total < min_positive_ratio:
            continue

        zeros = sub[sub[metric_col].isna() | (sub[metric_col] == 0)]
        for idx, row in zeros.iterrows():
            msg = (
                f"Aynı grup {group_cols} içinde kurumların {positive}/{total} tanesinde "
                f"{metric_col} > 0 iken bu satırda değer 0 veya boş."
            )
            issues.append(
                ValidationIssue(
                    severity=severity,
                    rule_code=rule_code,
                    message=msg,
                    row_index=int(idx),
                    context={
                        "group_key": key if isinstance(key, tuple) else (key,),
                        id_col: row.get(id_col),
                        metric_col: None,
                    },
                )
            )

    return issues


def v_high_outlier_generic(
    df: pd.DataFrame,
    group_cols: list[str],
    id_col: str,
    metric_col: str = "metrik_deger",
    min_group_size: int = 10,
    iqr_factor: float = 3.0,
    median_factor: float = 5.0,
    rule_code: str = "METRIC_OUTLIER_HIGH",
    severity: Severity = Severity.WARN,
) -> List[ValidationIssue]:
    """
    Aynı group_cols içinde metric_col değeri diğerlerine göre aşırı yüksekse
    WARN/FATAL üretir. IQR + median tabanlı eşik kullanır.
    """
    issues: List[ValidationIssue] = []

    required = set(group_cols + [id_col, metric_col])
    if not required.issubset(df.columns):
        return issues

    work = df.copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work[work[metric_col].notna()]
    if work.empty:
        return issues

    grp = work.groupby(group_cols, dropna=False)

    for key, sub in grp:
        if len(sub) < min_group_size:
            continue

        vals = sub[metric_col]
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue

        median = vals.median()
        threshold = q3 + iqr_factor * iqr
        threshold = max(threshold, median * median_factor)

        outliers = sub[vals > threshold]
        for idx, row in outliers.iterrows():
            val = float(row[metric_col])
            msg = (
                f"Aynı grup {group_cols} içindeki diğer değerlere göre bu satırın "
                f"{metric_col} değeri ({val}) olağan dışı derecede yüksek görünüyor."
            )
            issues.append(
                ValidationIssue(
                    severity=severity,
                    rule_code=rule_code,
                    message=msg,
                    row_index=int(idx),
                    context={
                        "group_key": key if isinstance(key, tuple) else (key,),
                        id_col: row.get(id_col),
                        metric_col: val,
                        "q1": float(q1),
                        "q3": float(q3),
                        "median": float(median),
                        "threshold": float(threshold),
                    },
                )
            )

    return issues


# ==========================================================
#  ACIL KURAL DEF (acil_kural_def) LOADER + PARAM PARSER
# ==========================================================

_rules_cache: dict[str, pd.DataFrame] = {}

def load_rules_by_category(kategori: str, refresh: bool = False) -> pd.DataFrame:
    """
    hastane_analiz.acil_kural_def tablosundaki aktif kuralları
    verilen kategori için çeker (ACIL, DOGUM, YOGUNBAKIM vb.).
    """
    kategori = kategori.upper()

    if not refresh and kategori in _rules_cache:
        return _rules_cache[kategori]

    sql = """
        SELECT
            alan_adi,
            gosterim_adi,
            metrik_yolu,
            kategori,
            sayfa_adi,
            veri_tipi,
            rol,
            kural_tipi,
            kural_param,
            severity,
            aktif_mi,
            aciklama
        FROM hastane_analiz.acil_kural_def
        WHERE aktif_mi = TRUE
          AND kategori = %s
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(kategori,))

    df["kural_tipi"] = df["kural_tipi"].str.upper().str.strip()
    df["severity"] = df["severity"].str.upper().str.strip()
    df["veri_tipi"] = df["veri_tipi"].str.lower().str.strip()
    df["rol"] = df["rol"].str.lower().str.strip()

    _rules_cache[kategori] = df
    return df



def parse_kural_param(param_str: Optional[str]) -> Dict[str, Any]:
    """
    'window=6;min=0;max=100' veya
    'allowed=0,1' gibi stringleri dict'e çevirir.
    """
    out: Dict[str, Any] = {}
    if not param_str:
        return out

    for part in str(param_str).split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "allowed":
            out[key] = [v.strip() for v in val.split(",") if v.strip() != ""]
        elif key in ("min", "max", "window"):
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = None
        else:
            out[key] = val
    return out


# ==========================================================
#  GENEL KURALLAR
# ==========================================================

def v_required_columns(df: pd.DataFrame,
                       required_cols: List[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for col in required_cols:
        if col not in df.columns:
            issues.append(
                ValidationIssue(
                    severity=Severity.FATAL,
                    rule_code="REQ_COL_MISSING",
                    message=f"Zorunlu kolon eksik: {col}",
                )
            )
    return issues


def v_year_month_range(
    df: pd.DataFrame,
    year_col: str = "yil",
    month_col: str = "ay",
    min_year: int = 2015,
    max_year: int = 2026,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    if year_col not in df.columns or month_col not in df.columns:
        return issues

    yil_ser = pd.to_numeric(df[year_col], errors="coerce")
    ay_ser = pd.to_numeric(df[month_col], errors="coerce")

    mask_bad_year = (yil_ser < min_year) | (yil_ser > max_year) | yil_ser.isna()
    for idx in df.index[mask_bad_year]:
        val = df.at[idx, year_col]
        issues.append(
            ValidationIssue(
                severity=Severity.FATAL,
                rule_code="YEAR_OUT_OF_RANGE",
                message=f"Yıl hatalı veya aralık dışında: {val!r}",
                row_index=int(idx),
                context={"yil_raw": str(val)},
            )
        )

    mask_bad_month = (ay_ser < 1) | (ay_ser > 12) | ay_ser.isna()
    for idx in df.index[mask_bad_month]:
        val = df.at[idx, month_col]
        issues.append(
            ValidationIssue(
                severity=Severity.FATAL,
                rule_code="MONTH_OUT_OF_RANGE",
                message=f"Ay 1-12 aralığında değil veya boş: {val!r}",
                row_index=int(idx),
                context={"ay_raw": str(val)},
            )
        )

    return issues


def v_metric_numeric(df: pd.DataFrame,
                     metric_col: str = "metrik_deger") -> List[ValidationIssue]:
    """
    metrik_deger sayıya çevrilemezse WARN üretir.
    """
    issues: List[ValidationIssue] = []

    if metric_col not in df.columns:
        return issues

    coerced = pd.to_numeric(df[metric_col], errors="coerce")
    mask_nan = coerced.isna() & df[metric_col].notna()

    for idx in df.index[mask_nan]:
        raw_val = df.at[idx, metric_col]
        issues.append(
            ValidationIssue(
                severity=Severity.WARN,
                rule_code="METRIC_NOT_NUMERIC",
                message=f"metrik_deger sayıya çevrilemedi: {raw_val!r}",
                row_index=int(idx),
                context={"raw_value": str(raw_val)},
            )
        )

    return issues


# ==========================================================
#  ACIL İÇİN YARDIMCI FONKSİYONLAR
# ==========================================================

def _ensure_numeric_copy(df: pd.DataFrame) -> pd.DataFrame:
    """
    ACIL kontrolleri için yardımcı:
    - yil, ay, metrik_deger'i numeric'e zorlar (hata olursa NaN).
    Orijinal df'i bozmaz, kopya döner.
    """
    work = df.copy()
    for col in ["yil", "ay"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "kurum_kodu" in work.columns:
        work["kurum_kodu"] = work["kurum_kodu"].astype(str)
    if "metrik_deger" in work.columns:
        work["metrik_deger"] = pd.to_numeric(work["metrik_deger"], errors="coerce")
    return work


def _infer_period_from_df(df: pd.DataFrame) -> Optional[tuple[int, int]]:
    """
    Dosyadaki yıl/ay bilgisinden tek dönem çıkarır.
    Ay ay yükleme senaryosunda her dosya için 1 yıl-1 ay bekliyoruz.
    """
    if "yil" not in df.columns or "ay" not in df.columns:
        return None

    yil_vals = pd.to_numeric(df["yil"], errors="coerce").dropna().unique()
    ay_vals = pd.to_numeric(df["ay"], errors="coerce").dropna().unique()

    if len(yil_vals) != 1 or len(ay_vals) != 1:
        return None

    return int(yil_vals[0]), int(ay_vals[0])


def _prev_period(yil: int, ay: int) -> tuple[int, int]:
    if ay > 1:
        return yil, ay - 1
    return yil - 1, 12




# ==========================================================
#  ACIL: KURAL TABANLI ENGINE (acil_kural_def)
# ==========================================================

def _normalize_bool_series(s: pd.Series) -> pd.Series:
    """
    Boolean flag'leri 0/1'e normalize eder.
    """
    s_str = s.astype(str).str.strip().str.lower()
    return (
        (~s.isna())
        & (s_str != "")
        & (s_str != "nan")
        & (s_str != "0")
        & (s_str != "yok")
        & (s_str != "hayir")
        & (s_str != "hayır")
    ).astype(int)

def _build_allowed_checker(allowed_raw: List[str]) -> Callable[[Any], bool]:
    """
    allowed listesine göre bir is_allowed(value) fonksiyonu döner.
    Hem '0'/'1' hem 0/1/0.0/1.0 gibi değerleri aynı görür.
    """
    # Önce numeric mod dene
    numeric_mode = True
    allowed_num: set[float] = set()
    for a in allowed_raw:
        try:
            allowed_num.add(float(a))
        except (TypeError, ValueError):
            numeric_mode = False
            break

    if numeric_mode and allowed_num:
        # Tüm allowed'lar sayıya döndüyse → numeric mod
        def is_allowed(val: Any) -> bool:
            if pd.isna(val):
                return False
            try:
                return float(val) in allowed_num
            except (TypeError, ValueError):
                return False

        return is_allowed

    # Aksi halde string mod
    allowed_str = {str(a).strip() for a in allowed_raw}

    def is_allowed(val: Any) -> bool:
        if pd.isna(val):
            return False
        return str(val).strip() in allowed_str

    return is_allowed


def apply_range_rule_long(
    rule_row: pd.Series,
    df: pd.DataFrame,
    file_path: str,
    kategori: str,
) -> List[ValidationIssue]:
    """
    RANGE kuralı, long form veri için:
    - metrik_adi == rule_row['metrik_yolu'] olan satırlarda metrik_deger'e bakar.
    """
    issues: List[ValidationIssue] = []

    if "metrik_adi" not in df.columns or "metrik_deger" not in df.columns:
        return issues

    metric_name = rule_row["metrik_yolu"]
    sub = df[df["metrik_adi"] == metric_name]
    if sub.empty:
        return issues

    params = parse_kural_param(rule_row.get("kural_param"))
    sev = Severity(rule_row["severity"])
    kategori_u = kategori.upper()
    rule_code = f"{kategori_u}.{rule_row['alan_adi']}.RANGE"

    # --- allowed set modu ---
    allowed_raw = params.get("allowed")
    if allowed_raw:
        is_allowed = _build_allowed_checker(allowed_raw)

        for idx, row in sub.iterrows():
            val = row["metrik_deger"]
            if is_allowed(val):
                continue

            msg = (
                f"{rule_row['gosterim_adi']} için izin verilen değerler "
                f"{allowed_raw!r}, ancak {val!r} görüldü."
            )
            issues.append(
                ValidationIssue(
                    severity=sev,
                    rule_code=rule_code,
                    message=msg,
                    file_path=file_path,
                    kategori=kategori_u,
                    sayfa_adi=rule_row.get("sayfa_adi"),
                    row_index=int(idx),
                    context={
                        "metrik_adi": metric_name,
                        "deger": val,
                        "allowed": allowed_raw,
                    },
                )
            )
        return issues

    # --- min/max modu ---
    ser_num = pd.to_numeric(sub["metrik_deger"], errors="coerce")
    min_v = params.get("min")
    max_v = params.get("max")

    if min_v is not None:
        mask = ser_num < min_v
        for idx, val in ser_num[mask].items():
            msg = (
                f"{rule_row['gosterim_adi']} değeri {val} < min {min_v}. "
                "Beklenen aralığın altında."
            )
            issues.append(
                ValidationIssue(
                    severity=sev,
                    rule_code=rule_code,
                    message=msg,
                    file_path=file_path,
                    kategori=kategori_u,
                    sayfa_adi=rule_row.get("sayfa_adi"),
                    row_index=int(idx),
                    context={
                        "metrik_adi": metric_name,
                        "deger": float(val),
                        "min": min_v,
                    },
                )
            )

    if max_v is not None:
        mask = ser_num > max_v
        for idx, val in ser_num[mask].items():
            msg = (
                f"{rule_row['gosterim_adi']} değeri {val} > max {max_v}. "
                "Beklenen aralığın üzerinde."
            )
            issues.append(
                ValidationIssue(
                    severity=sev,
                    rule_code=rule_code,
                    message=msg,
                    file_path=file_path,
                    kategori=kategori_u,
                    sayfa_adi=rule_row.get("sayfa_adi"),
                    row_index=int(idx),
                    context={
                        "metrik_adi": metric_name,
                        "deger": float(val),
                        "max": max_v,
                    },
                )
            )

    return issues


def apply_boolean_change_rule_db(
    rule_row: pd.Series,
    df: pd.DataFrame,
    file_path: str,
    yil: int,
    ay: int,
    kategori: str,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    required_cols = {"kurum_kodu", "metrik_adi", "metrik_deger"}
    if not required_cols.issubset(df.columns):
        return issues

    metric_name = rule_row["metrik_yolu"]
    sub = df[df["metrik_adi"] == metric_name].copy()
    if sub.empty:
        return issues

    sub["kurum_kodu"] = sub["kurum_kodu"].astype(str)
    birim_list = sub["kurum_kodu"].dropna().unique().tolist()
    if not birim_list:
        return issues

    prev_yil, prev_ay = _prev_period(yil, ay)
    kategori_u = kategori.upper()

    sql = """
        SELECT
            yil,
            ay,
            kurum_kodu::text AS kurum_kodu,
            metrik_deger_numeric
        FROM hastane_analiz.raw_veri
        WHERE kategori   = %s
          AND sayfa_adi  = %s
          AND metrik_adi = %s
          AND yil        = %s
          AND ay         = %s
          AND kurum_kodu::text = ANY(%s)
    """
    params = (
        kategori_u,
        rule_row.get("sayfa_adi") or kategori_u,
        metric_name,
        prev_yil,
        prev_ay,
        birim_list,
    )

    with get_connection() as conn:
        df_prev = pd.read_sql(sql, conn, params=params)

    if df_prev.empty:
        return issues

    df_prev["kurum_kodu"] = df_prev["kurum_kodu"].astype(str)
    df_prev["prev_value"] = _normalize_bool_series(df_prev["metrik_deger_numeric"])
    prev_map = dict(zip(df_prev["kurum_kodu"], df_prev["prev_value"]))

    sev = Severity(rule_row["severity"])
    rule_code = f"{kategori_u}.{rule_row['alan_adi']}.BOOLEAN_CHANGE"

    sub["cur_value"] = _normalize_bool_series(sub["metrik_deger"])

    for idx, row in sub.iterrows():
        kurum = row["kurum_kodu"]
        cur = int(row["cur_value"])
        prev = prev_map.get(kurum)
        if prev is None:
            continue

        if cur != int(prev):
            msg = (
                f"{kurum} için {yil}-{ay:02d} döneminde "
                f"{rule_row['gosterim_adi']} durumu değişmiş görünüyor "
                f"({int(prev)} → {cur})."
            )
            issues.append(
                ValidationIssue(
                    severity=sev,
                    rule_code=rule_code,
                    message=msg,
                    file_path=file_path,
                    kategori=kategori_u,
                    sayfa_adi=rule_row.get("sayfa_adi"),
                    row_index=int(idx),
                    context={
                        "birim_kodu": kurum,
                        "yil": yil,
                        "ay": ay,
                        "prev_yil": prev_yil,
                        "prev_ay": prev_ay,
                        "prev_value": int(prev),
                        "value": cur,
                        "metrik_adi": metric_name,
                    },
                )
            )

    return issues


def apply_ts_mean_rule(rule_row, df, file_path):
    """
    TS_MEAN: Geçmiş X ayın ortalaması ile karşılaştırma.

    Gerekli kural_param:
        window=<int>
    """
    issues = []

    metric = rule_row["metrik_yolu"]
    sev = Severity(rule_row["severity"])
    sayfa_adi = rule_row.get("sayfa_adi") or None

    params = parse_kural_param(rule_row.get("kural_param"))
    window = int(params.get("window", 6))

    # Bu dosyadaki metrik subseti
    cur = df[df["metrik_adi"] == metric].copy()
    if cur.empty:
        return issues

    # Dönemi bul
    period = _infer_period_from_df(df)
    if period is None:
        return issues

    yil, ay = period

    # --- DB'den geçmiş X ayı çek ---
    sql = """
        SELECT yil, ay, kurum_kodu::text AS kurum_kodu, metrik_deger_numeric
        FROM hastane_analiz.raw_veri
        WHERE metrik_adi = %s
          AND kategori = %s
          AND sayfa_adi = %s
          AND (yil * 12 + ay) < (%s * 12 + %s)
        ORDER BY yil DESC, ay DESC
        LIMIT %s;
    """
    params_sql = (
        metric,
        rule_row["kategori"],
        sayfa_adi,
        yil, ay,
        window,
    )

    with get_connection() as conn:
        prev_df = pd.read_sql(sql, conn, params=params_sql)

    if prev_df.empty:
        return issues

    prev_df["kurum_kodu"] = prev_df["kurum_kodu"].astype(str)
    prev_df["val"] = pd.to_numeric(prev_df["metrik_deger_numeric"], errors="coerce")
    means = prev_df.groupby("kurum_kodu")["val"].mean().to_dict()

    for idx, row in cur.iterrows():
        kurum = str(row["kurum_kodu"])
        cur_val = row["metrik_deger"]
        prev_mean = means.get(kurum)

        if prev_mean is None:
            continue

        # oran ≈ mevcut değerin 6 aylık ortalamaya göre farkı
        if prev_mean > 0:
            change_ratio = cur_val / prev_mean
        else:
            change_ratio = None

        # Eşik: çok gevşek
        if change_ratio is not None and (change_ratio > 3 or change_ratio < 0.25):
            msg = (
                f"{rule_row['gosterim_adi']} değeri geçmiş {window} ay ortalamasından "
                f"anlamlı derecede sapmış görünüyor: "
                f"mevcut={cur_val}, ortalama={prev_mean:.1f}, oran={change_ratio:.2f}"
            )
            issues.append(
                ValidationIssue(
                    severity=sev,
                    rule_code=f"TS_MEAN.{rule_row['alan_adi']}",
                    message=msg,
                    file_path=file_path,
                    kategori=rule_row["kategori"],
                    sayfa_adi=sayfa_adi,
                    row_index=int(idx),
                    context={
                        "cur_val": cur_val,
                        "prev_mean": prev_mean,
                        "ratio": change_ratio,
                    },
                )
            )

    return issues


def apply_sum_eq_rule(rule_row, df, file_path):
    """
    SUM_EQ: alt metriklerin toplamı bir hedef metriğe eşit olmalı.

    kural_param:
        group = m1,m2,m3
        target = toplam_metrik_adi
    """
    issues = []

    sev = Severity(rule_row["severity"])
    params = parse_kural_param(rule_row["kural_param"])

    group_metrics = params.get("group")
    target_metric = params.get("target")

    if not group_metrics or not target_metric:
        return issues

    group_list = [g.strip() for g in group_metrics.split(",")]

    # alt metrikler
    df_group = df[df["metrik_adi"].isin(group_list)].copy()
    df_target = df[df["metrik_adi"] == target_metric].copy()

    if df_group.empty or df_target.empty:
        return issues

    # kurum bazında grup toplamı
    gsum = df_group.groupby("kurum_kodu")["metrik_deger"].sum().to_dict()

    for idx, row in df_target.iterrows():
        kurum = row["kurum_kodu"]
        target_val = row["metrik_deger"]
        g_total = gsum.get(kurum)

        if g_total is None:
            continue

        if abs(g_total - target_val) > 0.01:
            msg = (
                f"{target_metric} değeri ({target_val}), alt metriklerin toplamına ({g_total}) eşit değil."
            )

            issues.append(
                ValidationIssue(
                    severity=sev,
                    rule_code=f"SUM_EQ.{rule_row['alan_adi']}",
                    message=msg,
                    file_path=file_path,
                    kategori=rule_row["kategori"],
                    sayfa_adi=rule_row.get("sayfa_adi"),
                    row_index=int(idx),
                    context={
                        "target": target_val,
                        "group_total": g_total,
                        "group_metrics": group_list,
                    },
                )
            )

    return issues





def run_rule_engine_for_category(
    df: pd.DataFrame,
    file_path: str,
    kategori: str,
    sayfa_adi: Optional[str],
) -> List[ValidationIssue]:
    """
    Kategori bağımsız kural motoru:
    - acil_kural_def + ileride dogum_kural_def vs. değil,
      tek bir generic kural tablosundan okuyoruz (rule_def gibi).
    """
    # Burada kategori bilgisini mutlaka geçiriyoruz:
    rules = load_rules_by_category(kategori)

    # Eğer sayfa_adi filtrelemek istersen:
    if sayfa_adi:
        rules = rules[(rules["sayfa_adi"] == sayfa_adi) | rules["sayfa_adi"].isna()]

    issues: List[ValidationIssue] = []

    # Burada kural_tipi'ne göre TS_MEAN / SUM_EQ / RATIO_RANGE vs. çağırıyoruz
    for _, rule in rules.iterrows():
        kural_tipi = rule["kural_tipi"]

        if kural_tipi == "TS_MEAN":
            issues.extend(apply_ts_mean_rule(rule, df, file_path))
        elif kural_tipi == "SUM_EQ":
            issues.extend(apply_sum_eq_rule(rule, df, file_path))
        elif kural_tipi == "RANGE":
            # kategori bilgisini de geçir
            issues.extend(apply_range_rule_long(rule, df, file_path, kategori))
        elif kural_tipi in ("BOOLEAN_CHANGE", "CHANGE"):
            period = _infer_period_from_df(df)
            if period is not None:
                yil, ay = period
                issues.extend(
                    apply_boolean_change_rule_db(
                        rule, df, file_path, yil, ay, kategori
                    )
                )
        

    return issues



# ==========================================================
#  ORKESTRASYON
# ==========================================================

def run_validations(
    df: pd.DataFrame,
    file_path: str,
    kategori: str,
    sayfa_adi: Optional[str] = None,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # 1) Zorunlu kolon kontrolü (tüm kategoriler)
    issues += v_required_columns(
        df,
        required_cols=["yil", "ay", "kurum_kodu", "metrik_adi", "metrik_deger"],
    )

    # 2) Zorunlu kolonlar tam ise genel kurallar
        # 2) Zorunlu kolonlar tam ise genel kurallar
    if not any(i.severity == Severity.FATAL for i in issues):
        issues += v_year_month_range(df)
        issues += v_metric_numeric(df)

        # 3) Kural motoru (TS_MEAN, RANGE, BOOLEAN_CHANGE, SUM_EQ, RATIO_RANGE...)
        issues += run_rule_engine_for_category(df, file_path, kategori, sayfa_adi)

        # 4) Heuristic kurallar (tüm kategoriler için geçerli)
        # Aynı (yil, ay, metrik_adi) grubunda kurum bazlı kontrol
        issues += v_zero_while_others_positive_generic(
            df,
            group_cols=["yil", "ay", "metrik_adi"],
            id_col="kurum_kodu",
        )

        issues += v_high_outlier_generic(
            df,
            group_cols=["yil", "ay", "metrik_adi"],
            id_col="kurum_kodu",
        )


    # 5) Ortak metadata
    for i in issues:
        i.file_path = file_path
        i.kategori = kategori
        i.sayfa_adi = sayfa_adi

    # 6) Hiç hata yoksa INFO: OK
    if len(issues) == 0:
        issues.append(
            ValidationIssue(
                severity=Severity.INFO,
                rule_code="OK",
                message="Dosya başarıyla geçti (hiçbir hata bulunamadı).",
                file_path=file_path,
                kategori=kategori,
                sayfa_adi=sayfa_adi,
                row_index=None,
                context={"durum": "başarılı"},
            )
        )

    return issues


def save_issues_to_db(issues: List[ValidationIssue]) -> None:
    if not issues:
        return

    sql = """
        INSERT INTO hastane_analiz.etl_kalite_sonuc (
            seviye,
            kural_kodu,
            mesaj,
            kaynak_dosya,
            kategori,
            sayfa_adi,
            row_index,
            context_json
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """

    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return str(obj)

    rows: List[tuple] = []
    for i in issues:
        rows.append(
            (
                i.severity.value,
                i.rule_code,
                i.message,
                i.file_path,
                i.kategori,
                i.sayfa_adi,
                i.row_index,
                json.dumps(i.context or {}, ensure_ascii=False, default=_json_default),
            )
        )

    batch_insert(sql, rows)
