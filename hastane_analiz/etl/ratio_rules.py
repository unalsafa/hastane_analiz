# hastane_analiz/etl/ratio_rules.py

import json
import pandas as pd

from hastane_analiz.db.connection import get_connection, batch_insert
from hastane_analiz.etl.validation import Severity, ValidationIssue  # mevcut dataclass'ı tekrar kullanıyoruz


def _load_active_ratio_rules() -> pd.DataFrame:
    sql = """
        SELECT
            kural_id,
            kural_kodu,
            kategori,
            seviye,
            num_metrik_kodu,
            den_metrik_kodu,
            min_oran,
            max_oran
        FROM hastane_analiz.ratio_kural_def
        WHERE aktif_mi = TRUE;
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def _fetch_ratio_data(num_code: str, den_code: str) -> pd.DataFrame:
    """
    v_fact_metrik üzerinden:
      (yil, ay, birim_key) bazında num/den ve ratio hesaplar.
    """
    sql = """
        SELECT
            n.yil,
            n.ay,
            b.birim_key,
            b.birim_adi,
            b.ilce_adi,
            n.toplam_deger AS num_deger,
            d.toplam_deger AS den_deger,
            CASE
                WHEN d.toplam_deger IS NULL OR d.toplam_deger = 0 THEN NULL
                ELSE n.toplam_deger / d.toplam_deger
            END AS oran
        FROM hastane_analiz.v_fact_metrik n
        JOIN hastane_analiz.v_fact_metrik d
          ON n.yil = d.yil
         AND n.ay  = d.ay
         AND n.birim_key = d.birim_key
        JOIN hastane_analiz.birim_def b
          ON n.birim_key = b.birim_key
        WHERE n.metrik_kodu = %(num)s
          AND d.metrik_kodu = %(den)s;
    """
    params = {"num": num_code, "den": den_code}
    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=params)


def run_ratio_checks() -> None:
    rules = _load_active_ratio_rules()
    if rules.empty:
        print("Aktif oran kuralı yok.")
        return

    all_issues: list[ValidationIssue] = []

    for _, rule in rules.iterrows():
        df = _fetch_ratio_data(
            rule["num_metrik_kodu"],
            rule["den_metrik_kodu"],
        )
        if df.empty:
            continue

        min_o = rule["min_oran"]
        max_o = rule["max_oran"]

        mask_low = (df["oran"].notna()) & (min_o is not None) & (df["oran"] < min_o)
        mask_high = (df["oran"].notna()) & (max_o is not None) & (df["oran"] > max_o)

        sev = Severity(str(rule["seviye"]).upper())
        kural_kodu = rule["kural_kodu"]

        for _, row in df[mask_low | mask_high].iterrows():
            durum = "altında" if row["oran"] < (min_o or 0) else "üstünde"
            msg = (
                f"{row['ilce_adi']} / {row['birim_adi']} için "
                f"{kural_kodu} oranı beklenen aralığın {durum}. "
                f"Oran = {row['oran']:.2f}, "
            )
            if min_o is not None:
                msg += f"min={min_o} "
            if max_o is not None:
                msg += f"max={max_o}"

            all_issues.append(
                ValidationIssue(
                    severity=sev,
                    rule_code=f"RATIO_RANGE.{kural_kodu}",
                    message=msg,
                    file_path=None,          # dosya bazlı değil
                    kategori=rule.get("kategori") or None,
                    sayfa_adi=None,
                    row_index=None,
                    context={
                        "kural_kodu": kural_kodu,
                        "yil": int(row["yil"]),
                        "ay": int(row["ay"]),
                        "birim_adi": row["birim_adi"],
                        "ilce_adi": row["ilce_adi"],
                        "num_metrik_kodu": rule["num_metrik_kodu"],
                        "den_metrik_kodu": rule["den_metrik_kodu"],
                        "num_deger": float(row["num_deger"]) if row["num_deger"] is not None else None,
                        "den_deger": float(row["den_deger"]) if row["den_deger"] is not None else None,
                        "oran": float(row["oran"]) if row["oran"] is not None else None,
                        "min_oran": float(min_o) if min_o is not None else None,
                        "max_oran": float(max_o) if max_o is not None else None,
                    },
                )
            )

    # Sonuçları etl_kalite_sonuc'a yaz
    if all_issues:
        from hastane_analiz.etl.validation import save_issues_to_db
        save_issues_to_db(all_issues)
        print(f"{len(all_issues)} oran anomalisi kaydedildi.")
    else:
        print("Oran anomalisi bulunmadı.")


if __name__ == "__main__":
    run_ratio_checks()
