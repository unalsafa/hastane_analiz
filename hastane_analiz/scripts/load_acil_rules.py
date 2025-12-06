import pandas as pd
from pathlib import Path

from hastane_analiz.db.connection import get_connection  # sende zaten var
from psycopg2.extras import execute_batch


EXCEL_PATH = Path(r"C:\hastane_analiz\hastane_analiz\config\valid_acil.xlsx")  # kendi yoluna göre değiştir


def normalize_bool(val):
    if isinstance(val, bool):
        return val
    if val is None or pd.isna(val):
        return False
    return str(val).strip().lower() in ("1", "true", "evet", "yes")


def load_acil_rules():
    df = pd.read_excel(EXCEL_PATH)

    # Bazı temizlikler
    df["kategori"] = df["kategori"].fillna("ACIL").str.upper()
    df["sayfa_adi"] = df["sayfa_adi"].fillna("ACIL").str.upper()
    df["veri_tipi"] = df["veri_tipi"].str.lower().str.strip()
    df["rol"] = df["rol"].str.lower().str.strip()
    df["kural_tipi"] = df["kural_tipi"].str.upper().str.strip()
    df["severity"] = df["severity"].str.upper().str.strip()
    df["aktif_mi"] = df["aktif_mi"].apply(normalize_bool)

    rows = [
        (
            row["alan_adi"],
            row["gosterim_adi"],
            row["metrik_yolu"],
            row["kategori"],
            row["sayfa_adi"],
            row["veri_tipi"],
            row["rol"],
            row["kural_tipi"],
            row.get("kural_param"),
            row["severity"],
            row["aktif_mi"],
            row.get("aciklama"),
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO hastane_analiz.acil_kural_def (
        alan_adi, gosterim_adi, metrik_yolu, kategori, sayfa_adi,
        veri_tipi, rol, kural_tipi, kural_param, severity,
        aktif_mi, aciklama
    ) VALUES (
        %s,%s,%s,%s,%s,
        %s,%s,%s,%s,%s,
        %s,%s
    )
    ON CONFLICT (kategori, sayfa_adi, alan_adi, kural_tipi)
    DO UPDATE SET
        gosterim_adi   = EXCLUDED.gosterim_adi,
        metrik_yolu    = EXCLUDED.metrik_yolu,
        veri_tipi      = EXCLUDED.veri_tipi,
        rol            = EXCLUDED.rol,
        kural_param    = EXCLUDED.kural_param,
        severity       = EXCLUDED.severity,
        aktif_mi       = EXCLUDED.aktif_mi,
        aciklama       = EXCLUDED.aciklama,
        son_guncelleme = now();
    """

    with get_connection() as conn, conn.cursor() as cur:
        execute_batch(cur, sql, rows, page_size=100)
        conn.commit()
        print(f"{len(rows)} kural kaydı yüklendi / güncellendi.")


if __name__ == "__main__":
    load_acil_rules()
