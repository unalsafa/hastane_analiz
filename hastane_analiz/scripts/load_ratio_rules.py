import pandas as pd
from pathlib import Path

from hastane_analiz.db.connection import get_connection, batch_insert  




def load_ratio_rules_from_excel():
    base_dir = Path(__file__).resolve().parents[1]
    path = base_dir / "config" / "valid_ratio.xlsx"
    df = pd.read_excel(path, sheet_name="RATIO_RULES")

    print(f"[RATIO_RULES] Excel dosyası: {path}")

    # Boş/kapatılmış kuralları at
    df = df[df.get("aktif_mi", True).astype(bool)]

    rows = []
    for _, row in df.iterrows():
        rows.append(
            (
                row["kural_kodu"],
                row.get("kategori"),
                row.get("aciklama"),
                str(row.get("seviye") or "WARN").upper(),
                bool(row.get("aktif_mi", True)),
                row["num_metrik_kodu"],
                row["den_metrik_kodu"],
                row.get("min_oran"),
                row.get("max_oran"),
            )
        )

    sql = """
        INSERT INTO hastane_analiz.ratio_kural_def (
            kural_kodu,
            kategori,
            aciklama,
            seviye,
            aktif_mi,
            num_metrik_kodu,
            den_metrik_kodu,
            min_oran,
            max_oran
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (kural_kodu) DO UPDATE SET
            kategori        = EXCLUDED.kategori,
            aciklama        = EXCLUDED.aciklama,
            seviye          = EXCLUDED.seviye,
            aktif_mi        = EXCLUDED.aktif_mi,
            num_metrik_kodu = EXCLUDED.num_metrik_kodu,
            den_metrik_kodu = EXCLUDED.den_metrik_kodu,
            min_oran        = EXCLUDED.min_oran,
            max_oran        = EXCLUDED.max_oran;
    """

    with get_connection() as conn:
        batch_insert(sql, rows)

    print(f"{len(rows)} adet oran kuralı yüklendi/güncellendi.")


if __name__ == "__main__":
    load_ratio_rules_from_excel()
