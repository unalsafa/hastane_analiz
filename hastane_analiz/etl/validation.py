from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Callable

import json
import pandas as pd

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
#  ACIL KURAL DEF (acil_kural_def) LOADER + PARAM PARSER
# ==========================================================

_acil_rules_cache: Optional[pd.DataFrame] = None


def load_acil_rules(refresh: bool = False) -> pd.DataFrame:
    """
    hastane_analiz.acil_kural_def tablosundaki aktif kuralları çeker.
    """
    global _acil_rules_cache
    if _acil_rules_cache is not None and not refresh:
        return _acil_rules_cache

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
          AND kategori = 'ACIL'
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn)

    # küçük normalizasyon
    df["kural_tipi"] = df["kural_tipi"].str.upper().str.strip()
    df["severity"] = df["severity"].str.upper().str.strip()
    df["veri_tipi"] = df["veri_tipi"].str.lower().str.strip()
    df["rol"] = df["rol"].str.lower().str.strip()

    _acil_rules_cache = df
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
    max_year: int = 2035,
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
#  ACIL: DOSYA İÇİ HEURISTIC KURALLAR (outlier vs.)
# ==========================================================

def v_zero_while_others_positive_acil(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Aynı (yil, ay, metrik_adi) için kurumların büyük çoğunluğu > 0 iken
    değeri 0 veya NaN olan satırlar için WARN üretir.

    "Herkes yapmış, bu kurum hiç yapmamış" tipinde uyarı.
    """
    issues: List[ValidationIssue] = []
    required = {"yil", "ay", "kurum_kodu", "metrik_adi", "metrik_deger"}
    if not required.issubset(df.columns):
        return issues

    work = _ensure_numeric_copy(df)

    grp = work.groupby(["yil", "ay", "metrik_adi"], dropna=False)
    for (yil, ay, metrik_adi), sub in grp:
        if len(sub) < 5:
            continue  # çok az kurum varsa güvenilir değil

        total = len(sub)
        positive = (sub["metrik_deger"] > 0).sum()
        if positive / total < 0.7:
            # Çoğunluk zaten 0/boş, buradan anlamlı uyarı çıkmaz
            continue

        zeros = sub[(sub["metrik_deger"].isna()) | (sub["metrik_deger"] == 0)]
        for idx, row in zeros.iterrows():
            msg = (
                f"Aynı dönem ve metrikte kurumların {positive}/{total} tanesinde "
                f"değer > 0 iken bu satırda değer 0 veya boş."
            )
            issues.append(
                ValidationIssue(
                    severity=Severity.WARN,
                    rule_code="ZERO_WHILE_OTHERS_POSITIVE",
                    message=msg,
                    row_index=int(idx),
                    context={
                        "yil": int(yil) if pd.notna(yil) else None,
                        "ay": int(ay) if pd.notna(ay) else None,
                        "kurum_kodu": row.get("kurum_kodu"),
                        "metrik_adi": row.get("metrik_adi"),
                        "metrik_deger": None,
                    },
                )
            )

    return issues


def v_high_outlier_acil(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Aynı (yil, ay, metrik_adi) grubunda metrik_deger diğerlerine göre
    aşırı yüksekse WARN üretir.

    Basit bir IQR (Q3 + 3*IQR) + medyan*5 eşiği kullanıyoruz.
    Bu kurallar ileride ML tabanlı anomaly detection için etiket görevi görebilir.
    """
    issues: List[ValidationIssue] = []
    required = {"yil", "ay", "kurum_kodu", "metrik_adi", "metrik_deger"}
    if not required.issubset(df.columns):
        return issues

    work = _ensure_numeric_copy(df)
    work = work[work["metrik_deger"].notna()]
    if work.empty:
        return issues

    grp = work.groupby(["yil", "ay", "metrik_adi"], dropna=False)

    for (yil, ay, metrik_adi), sub in grp:
        if len(sub) < 10:
            continue  # istatistiksel anlam için minimum

        vals = sub["metrik_deger"]
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue

        median = vals.median()
        threshold = q3 + 3 * iqr
        threshold = max(threshold, median * 5)  # çok küçük metriklerde şişmesin

        outliers = sub[vals > threshold]
        for idx, row in outliers.iterrows():
            val = float(row["metrik_deger"])
            msg = (
                f"Aynı dönem ve metrikteki diğer değerlere göre bu satırın değeri "
                f"({val}) olağan dışı derecede yüksek görünüyor."
            )
            issues.append(
                ValidationIssue(
                    severity=Severity.WARN,
                    rule_code="METRIC_OUTLIER_HIGH",
                    message=msg,
                    row_index=int(idx),
                    context={
                        "yil": int(yil) if pd.notna(yil) else None,
                        "ay": int(ay) if pd.notna(ay) else None,
                        "kurum_kodu": row.get("kurum_kodu"),
                        "metrik_adi": row.get("metrik_adi"),
                        "metrik_deger": val,
                        "q1": float(q1),
                        "q3": float(q3),
                        "median": float(median),
                        "threshold": float(threshold),
                    },
                )
            )

    return issues


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


def apply_range_rule_long(rule_row: pd.Series,
                          df: pd.DataFrame,
                          file_path: str) -> List[ValidationIssue]:
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
    rule_code = f"ACIL.{rule_row['alan_adi']}.RANGE"

    allowed_raw = params.get("allowed")
    if allowed_raw:
        # allowed=0,1 → hem 0 / 1 / 0.0 / "0" / "1" hepsini aynı gör
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
                    kategori="ACIL",
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

    # allowed yoksa min/max moduna geç
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
                    kategori="ACIL",
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
                    kategori="ACIL",
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
) -> List[ValidationIssue]:
    """
    BOOLEAN_CHANGE / CHANGE:
    - Bu dosyadaki değerleri, raw_veri'de bir önceki aya ait değerlerle kıyaslar.
    - long form: metrik_adi == rule_row['metrik_yolu'], metrik_deger -> bool
    """
    issues: List[ValidationIssue] = []

    required_cols = {"kurum_kodu", "metrik_adi", "metrik_deger"}
    if not required_cols.issubset(df.columns):
        return issues

    metric_name = rule_row["metrik_yolu"]
    sub = df[df["metrik_adi"] == metric_name].copy()
    if sub.empty:
        return issues

    # Bu dosyada geçen birimler
    sub["kurum_kodu"] = sub["kurum_kodu"].astype(str)
    birim_list = sub["kurum_kodu"].dropna().unique().tolist()
    if not birim_list:
        return issues

    prev_yil, prev_ay = _prev_period(yil, ay)

    # DB'den önceki ayın değerleri
    sql = """
        SELECT
            yil,
            ay,
            kurum_kodu::text AS kurum_kodu,
            metrik_deger_numeric
        FROM hastane_analiz.raw_veri
        WHERE kategori   = 'ACIL'
          AND sayfa_adi  = %s
          AND metrik_adi = %s
          AND yil        = %s
          AND ay         = %s
          AND kurum_kodu::text = ANY(%s)
    """
    params = (
        rule_row.get("sayfa_adi") or "ACIL",
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
    rule_code = f"ACIL.{rule_row['alan_adi']}.BOOLEAN_CHANGE"

    sub["cur_value"] = _normalize_bool_series(sub["metrik_deger"])

    for idx, row in sub.iterrows():
        kurum = row["kurum_kodu"]
        cur = int(row["cur_value"])
        prev = prev_map.get(kurum)
        if prev is None:
            continue  # önceki ayda kayıt yoksa şimdilik sessiz geçiyoruz

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
                    kategori="ACIL",
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


def run_acil_rule_engine(
    df: pd.DataFrame,
    file_path: str,
    kategori: str,
    sayfa_adi: Optional[str],
) -> List[ValidationIssue]:
    """
    valid_acil.xlsx → acil_kural_def → buradan RANGE / BOOLEAN_CHANGE vb. uygular.
    """
    if kategori.upper() != "ACIL":
        return []

    rules = load_acil_rules()
    issues: List[ValidationIssue] = []

    # Aynı anda çoklu sayfa kullanırsak sayfa_adi ile filtreleyebiliriz
    if sayfa_adi:
        rules_use = rules[(rules["sayfa_adi"] == sayfa_adi)]
    else:
        rules_use = rules

    period = _infer_period_from_df(df)
    yil_ay_ok = period is not None
    if yil_ay_ok:
        cur_yil, cur_ay = period
    else:
        cur_yil = cur_ay = None  # BOOLEAN_CHANGE çalışmaz

    for _, rule in rules_use.iterrows():
        kural_tipi = rule["kural_tipi"]

        if kural_tipi == "RANGE":
            issues.extend(apply_range_rule_long(rule, df, file_path))

        elif kural_tipi in ("BOOLEAN_CHANGE", "CHANGE") and yil_ay_ok:
            issues.extend(apply_boolean_change_rule_db(rule, df, file_path, cur_yil, cur_ay))

        # ileride: TS_MEAN, SUM_EQ vb. burada ele alınacak

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
    if not any(i.severity == Severity.FATAL for i in issues):
        issues += v_year_month_range(df)
        issues += v_metric_numeric(df)

        # 3) ACIL'e özel kural motoru (acil_kural_def)
        if kategori.upper() == "ACIL":
            issues += run_acil_rule_engine(df, file_path, kategori, sayfa_adi)

            # 4) ACIL heuristic kurallar (opsiyonel, çok uyarı üretebilir)
            issues += v_zero_while_others_positive_acil(df)
            issues += v_high_outlier_acil(df)

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
                json.dumps(i.context or {}, ensure_ascii=False),
            )
        )

    batch_insert(sql, rows)
